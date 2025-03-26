import ezkl
import os

from config import Config
from responses import VerificationResult


class Verifier:
    _config: Config

    def __init__(self, config):
        self._config = config

    def exist_proof(self, proof_path) -> bool:
        return os.path.isfile(proof_path)

    def verify(self, id: str) -> VerificationResult:
        proof_path = f"{self._config.proofs_dir_path}/{id}.json"
        exist = self.exist_proof(proof_path)
        if not exist:
            return VerificationResult(id, False, "proof doesn't exist")

        try:

            print(f"proof to load: {proof_path}")
            verified = ezkl.verify(
                proof_path=proof_path,
                settings_path=self._config.settings_path,
                vk_path=self._config.verification_key_path,
                srs_path=self._config.srs_path,
            )
            return VerificationResult(id, verified)
        except Exception as e:
            print(e)
            return VerificationResult(id, False, e.__str__())

