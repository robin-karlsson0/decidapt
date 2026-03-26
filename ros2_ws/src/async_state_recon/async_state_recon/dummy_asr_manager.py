from typing import Any, Optional

from exodapt_robot_interfaces.srv import StartReconciliation

from .base_asr_manager import BaseASRManager
from .inference_client import ClientType, create_inference_client


class DummyASRManager(BaseASRManager):
    """Synchronous pass-through for single-GPU environments."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.declare_parameter('r1_url', 'http://localhost:8001')
        r1_url = self.get_parameter('r1_url').value

        # Initialize single client
        if self.client_type_str == 'vllm':
            client_type = ClientType.VLLM
        else:
            raise ValueError(
                f"Unsupported client_type: {self.client_type_str!r}")

        self._r_primary = create_inference_client(
            client_type,
            model_name=self.model_name,
            url=r1_url,
            name='R1',
        )

        # Dummy Service (Acknowledges evictions, does nothing)
        self._recon_service = self.create_service(
            StartReconciliation, self.recon_srv_name,
            self._dummy_reconciliation_callback)
        self.get_logger().info('DummyASRManager ready (Synchronous Mode).')

    def _dummy_reconciliation_callback(
        self,
        request: StartReconciliation.Request,
        response: StartReconciliation.Response,
    ) -> StartReconciliation.Response:
        self.get_logger().debug(
            f'Acknowledged eviction k={request.evicted_state_seq_ver}.')
        response.success = True
        return response

    def run(
        self,
        state_json_str: str,
        max_tokens: int = 512,
        temp: float = 0.7,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> Any:
        meta = self._parse_state(state_json_str)
        self.stats.total_inference_requests += 1
        # Direct pass-through
        return self._r_primary.run(meta.sequence,
                                   max_tokens=max_tokens,
                                   temp=temp,
                                   seed=seed,
                                   stream=stream)

    def get_status(self) -> dict:
        return {
            'mode': 'single_resource_sync',
            'stats': {
                'total_requests': self.stats.total_inference_requests
            }
        }
