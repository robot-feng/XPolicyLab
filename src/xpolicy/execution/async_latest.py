"""Single-flight prediction with one replaceable latest observation.

Reset invalidates local execution immediately. The worker finishes the current
model call, performs the newest required model reset, then predicts again.
It never overlaps model reset with inference or cancels a running vendor call.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from xpolicy.core.contracts import ActionChunk, ContractValidator, Observation
from xpolicy.execution.buffer import ActionBuffer, tick


class LocalPolicyBackend:
    """Run an existing synchronous runtime on a dedicated serial worker."""

    def __init__(self, policy):
        self.policy = policy
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xpolicy-inference")
        self._active_hook = None

    async def reset(self, episode_id):
        reset = getattr(self.policy, "reset_episode", None)
        await asyncio.wrap_future(self._executor.submit(reset, episode_id) if callable(reset) else self._executor.submit(self.policy.reset))

    async def predict(self, observation, request):
        return await asyncio.wrap_future(self._executor.submit(self.policy.predict_action_chunk, observation))

    async def prepare_case(self, case_meta):
        self._active_hook = self._executor.submit(self.policy.prepare_case, case_meta)
        return await asyncio.shield(asyncio.wrap_future(self._active_hook))

    async def on_trial_end(self, result):
        self._active_hook = self._executor.submit(self.policy.on_trial_end, result)
        return await asyncio.shield(asyncio.wrap_future(self._active_hook))

    @property
    def hook_pending(self):
        return self._active_hook is not None and not self._active_hook.done()

    def close(self):
        # Close is queued after running inference; it cannot race with it.
        # wait=False does not terminate a hung vendor call or its thread.
        future = self._executor.submit(self.policy.close)
        self._executor.shutdown(wait=False)
        return future


class WebSocketPolicyBackend:
    """Reuse PolicyEvalClient CALL/RESET and its existing retry/restart rules."""

    def __init__(self, client):
        self.client = client

    async def reset(self, episode_id):
        await self.client.reset(trial_id=episode_id)

    async def predict(self, observation, request):
        frame = await self.client.call("predict_action_chunk", observation.to_wire_dict(), trial_id=observation.episode_id,
                                       step=request.sequence_id, request_id=request.request_id)
        return ActionChunk.from_wire_dict(frame.payload["result"])


class AsyncLatestOnly:
    def __init__(self, backend, contracts, settings, *, emit=None, control_clock=None):
        if settings.strategy != "async_latest_only":
            raise ValueError("AsyncLatestOnly requires strategy=async_latest_only")
        self.backend, self.contracts, self.emit = backend, contracts, emit
        self.control_clock = control_clock
        self._closed, self._stopped = False, True
        self.buffer = ActionBuffer(contracts.manifest, settings, emit=self._emit)
        self._worker, self._latest = None, None
        self._applied_generation, self._last_frame_id = -1, -1
        self._now_s, self._clock_domain = None, None
        self.fault = None
        self.predictions, self.superseded_observations = 0, 0

    def _emit(self, value):
        if self.emit is not None and not self._closed:
            self.emit(value)

    def _ensure_worker(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work())

    def reset(self, episode_id: str, *, start_tick=0):
        if self._closed:
            raise RuntimeError("execution controller is closed")
        # There is no await before invalidation: callers cannot consume an
        # old-generation action while waiting for a remote model reset.
        generation = self.buffer.reset(episode_id, start_tick=start_tick)
        self._latest, self.fault = None, None
        self._last_frame_id, self._applied_generation = -1, -1
        self._now_s, self._clock_domain = None, None
        self.predictions, self.superseded_observations = 0, 0
        self._stopped = False
        self._ensure_worker()
        return generation

    def offer(self, observation: Observation, *, control_tick: int):
        if self._closed or self._stopped or self.fault is not None:
            raise RuntimeError("execution requires an active episode without a fault")
        control_tick = tick(control_tick)
        ContractValidator.observation(observation, self.contracts.robot, self.contracts.manifest)
        if observation.episode_id != self.buffer.episode_id:
            raise ValueError("observation episode differs from the active execution generation")
        if observation.frame_id <= self._last_frame_id:
            raise ValueError("observation frame IDs must increase within an episode")
        self.buffer.advance(max(self.buffer.current_tick, control_tick))
        self._last_frame_id = observation.frame_id
        self._now_s, self._clock_domain = observation.timestamp_s, observation.clock_domain
        if self._latest is not None:
            self.superseded_observations += 1
            self._emit({"stage": "observation_superseded", "frame_id": self._latest[0].frame_id,
                        "replacement_frame_id": observation.frame_id, "episode_generation": self.buffer.generation,
                        "episode_id": observation.episode_id, "control_tick": control_tick})
        self._latest = observation, control_tick
        self._ensure_worker()

    def refresh_clock(self):
        if self.control_clock is not None:
            control_tick, now_s, clock_domain = self.control_clock()
            self.buffer.advance(control_tick)
            self._now_s, self._clock_domain = now_s, clock_domain

    def take(self, control_tick: int, *, now_s=None, clock_domain=None):
        if self._closed or self._stopped or self.fault is not None:
            return None
        return self.buffer.take(control_tick, now_s=self._now_s if now_s is None else now_s,
                                clock_domain=self._clock_domain if clock_domain is None else clock_domain)

    def stop(self, reason="stopped"):
        self._stopped = True
        self._latest = None
        self.buffer.clear(reason)

    def fail(self, exc):
        self.fault = {"type": type(exc).__name__, "message": str(exc)}
        self.stop("fault")
        self._emit({"stage": "execution_fault", "episode_generation": self.buffer.generation,
                    "episode_id": self.buffer.episode_id, "fault": self.fault})

    async def _work(self):
        while not self._closed and not self._stopped:
            generation = self.buffer.generation
            try:
                if self._applied_generation != generation:
                    episode_id = self.buffer.episode_id
                    await self.backend.reset(episode_id)
                    if generation != self.buffer.generation:
                        continue
                    self._applied_generation = generation
                    self.buffer.event("model_reset_complete")
                if self._closed or self._stopped or self._latest is None:
                    return
                observation, source_tick = self._latest
                self._latest = None
                self.refresh_clock()
                request = self.buffer.issue(observation, control_tick=source_tick)
                started = time.monotonic()
                try:
                    chunk = await self.backend.predict(observation, request)
                except Exception:
                    self.buffer.event("inference_timing", request, duration_s=time.monotonic() - started, outcome="error")
                    raise
                self.buffer.event("inference_timing", request, duration_s=time.monotonic() - started, outcome="returned")
                if generation != self.buffer.generation:
                    self.buffer.event("prediction_discard", request, reason="old_generation")
                    continue
                if self._stopped or self._closed:
                    self.buffer.event("prediction_discard", request, reason="execution_stopped")
                    return
                self.refresh_clock()
                if self._stopped or self._closed:
                    self.buffer.event("prediction_discard", request, reason="execution_stopped")
                    return
                self.predictions += 1
                if not isinstance(chunk, ActionChunk):
                    raise TypeError("prediction must be an ActionChunk")
                self.buffer.event("prediction", request, trace_layer="A0", values=chunk.values.tolist(),
                                  shape=chunk.values.shape, action_type=chunk.action_type.value,
                                  action_units=chunk.action_units, joint_names=chunk.joint_names,
                                  latency_s=time.monotonic() - started)
                self.buffer.accept(request, chunk, control_tick=self.buffer.current_tick,
                                   now_s=self._now_s, clock_domain=self._clock_domain)
            except Exception as exc:
                if generation != self.buffer.generation:
                    self.buffer.event("obsolete_inference_error", error={"type": type(exc).__name__, "message": str(exc)}, old_generation=generation)
                    continue
                if not self._closed:
                    self.fail(exc)
                return

    async def wait_idle(self, timeout_s=None):
        if self._worker is not None:
            # A timeout invalidates scheduling via fail(), but never cancels
            # an in-flight model call or releases its serial ownership early.
            try:
                await asyncio.wait_for(asyncio.shield(self._worker), timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                self.fail(TimeoutError("inference worker did not become idle before its deadline"))
                raise TimeoutError(self.fault["message"]) from exc

    async def close(self, *, timeout_s=None):
        self.stop("closed")
        self._closed = True
        if self._worker is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._worker), timeout=timeout_s)
                return True
            except asyncio.TimeoutError:
                return False
        return True
