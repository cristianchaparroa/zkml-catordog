from content import ImageContent
from config import Config
import ezkl
import json
from common.logger import new_logger


class ImageProver:
    _conf: Config

    def __init__(self, conf):
        self._conf = conf
        self._log = new_logger(__name__)

    async def generate_proof(self, i: ImageContent):
        witness_path = await self.generate_witness(i)
        output_path = f"{self._conf.proofs_dir_path}/{i.get_id()}.json"

        # Directly await the future
        result = await ezkl.prove(
            witness=witness_path,
            model=self._conf.circuit_path,
            pk_path=self._conf.proving_key_path,
            srs_path=self._conf.srs_path,
            proof_path=output_path
        )

        self._log.info(f"proof generated for id:{i.get_id()}")
        return output_path

    async def write_input_date(self, i: ImageContent):
        input_data = i.get_witness_input()
        output_path = f"{self._conf.inputs_dir_path}/{i.get_id()}.json"

        with open(output_path, 'w') as f:
            json.dump(input_data, f)

        return output_path

    async def generate_witness(self, i: ImageContent):
        input_data_path = await self.write_input_date(i)
        output_path = f"{self._conf.witness_dir_path}/{i.get_id()}.json"

        # Directly await the future
        result = await ezkl.gen_witness(
            data=input_data_path,
            model=self._conf.circuit_path,
            vk_path=self._conf.verification_key_path,
            srs_path=self._conf.srs_path,
            output=output_path
        )

        self._log.info(f"witness generated for id:{i.get_id()}")
        return output_path
