from typing import Iterator, Tuple, Any

import glob
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os

class RoboverseDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Polybot."""

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
      '1.0.1': 'Multi-Object Perspective Grasp',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format=None,
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(10,),
                            dtype=np.float64,
                            doc='Robot state (joint angles)',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=
                '/iris/u/jyang27/dev/roboverse/data/pickplace_perspective'),
        }


    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, idx):
            # load raw data --> this should change for your dataset
            data = episodes[idx]
            episode_length = len(data)

            episode = []
            for i in range(episode_length):
                obs = data['observations'][i]
                language_instruction = 'Perform a task.'
                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()

                image = obs['image']
                state = obs['state'] 

                action = data['actions'][i][:7] #Get rid of "neutral" action in last dim

                episode.append({
                    'observation': {
                        'image': image,
                        'state': state,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (episode_length - 1)),
                    'is_first': i == 0,
                    'is_last': i == (episode_length - 1),
                    'is_terminal': i == (episode_length - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample
        
        
        globpath = Path(path)
        paths = list(globpath.rglob('*.npy')) 
        
        for p in paths:
            episodes = np.load(p, allow_pickle=True)
            for i in range(len(episodes)):
                yield _parse_example(os.path.join(p, f'{i}'), i)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

