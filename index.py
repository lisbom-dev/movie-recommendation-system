import os

from absl import logging
from tfx import v1 as tfx

PIPELINE_NAME = 'TFRS-movie-ranking'

DATA_ROOT = os.path.join('data', PIPELINE_NAME)
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join('pipelines', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

logging.set_verbosity(logging.INFO)

_trainer_module_file = 'ranking_model_trainer.py'

def _create_pipeline(pipeline_name: str, pipeline_root: str,
                     data_root: str, module_file: str,
                     serving_model_dir: str, metadata_path: str) -> tfx.dsl.Pipeline:

    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=12),
        eval_args=tfx.proto.EvalArgs(num_steps=24)
    )

    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    components = [
        example_gen,
        trainer,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
        .sqlite_metadata_connection_config(metadata_path),
        components=components
    )


if __name__ == '__main__':
    tfx.orchestration.LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_ROOT,
            module_file=_trainer_module_file,
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_path=METADATA_PATH
        )
    )
