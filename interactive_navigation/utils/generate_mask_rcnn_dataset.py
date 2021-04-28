# generate the metadata for alfred tasks.
import ai2thor
from typing import Dict, Tuple, List, Any, Optional, Union, Sequence, cast
from interactive_navigation.environment import IThorEnvironment
import interactive_navigation.constants as constants
import json
import pdb
import random
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor
import numpy as np
import gzip
import os
from functools import reduce #python 3
import operator
import multiprocessing as mp
import argparse
from torchvision.utils import save_image
import PIL
import pickle
import time

from interactive_navigation.task_samplers import ObstaclesNavDatasetTaskSampler


OBSTACLES_TYPES = ["ArmChair", "DogBed", "Box", "Chair", "Desk", "DiningTable", "SideTable", "Sofa",
                   "Stool", "Television", "Pillow", "Bread", "Apple", "AlarmClock", "Lettuce",
                   "GarbageCan", "Laptop", "Microwave", "Pot", "Tomato"]

OBJECT_TYPES = sorted(list(OBSTACLES_TYPES))

OBJECT_TYPES_TO_ID = {object_type:i+1 for i, object_type in enumerate(OBJECT_TYPES)}

SCENE_NAMES = constants.TRAIN_SCENE_NAMES

TRAIN_DATASET_DIR = "datasets/ithor-pointnav-obstacles_v2/train"


def main(args, thread_num=0):

    gpu_ids = args.gpu_ids
    
    x_display = "0.%d" %(thread_num % len(gpu_ids))

    floor_plan = SCENE_NAMES[thread_num % len(SCENE_NAMES)]
    print("generating room %s" %floor_plan)
    
    env = IThorEnvironment(
        x_display=x_display,
        player_screen_width=224,
        player_screen_height=224,
    )
    env.reset(scene_name=floor_plan)

    episodes = ObstaclesNavDatasetTaskSampler.load_dataset(
        floor_plan, TRAIN_DATASET_DIR + "/episodes"
    )

    count = 0
    for i in range(args.image_num):
        if i > len(episodes):
            episode_id = i - len(episodes)
        else:
            episode_id = i
        episode = episodes[episode_id]

        env.reset(scene_name=episode["scene"])
        env.teleport(
            pose=episode["initial_position"], rotation=episode["initial_orientation"]
        )
        for obj in episode["spawn_objects"]:
            env.spawn_obj(obj)
        env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)
        event = env.last_event

        object_types_in_scene = set([o["objectType"] for o in event.metadata["objects"]])
        for object_type in object_types_in_scene:
            if object_type in OBJECT_TYPES:
                # teleport to the scene that the object is visible.
                y = env.last_event.metadata["agent"]["position"]["y"]
                positions_to_check_interactionable_from = [
                    {"x": x, "y": y, "z": z}
                    for x, z in set((x, z) for x, z, _, _ in env.graph.nodes)
                ]
                if len(env.get_objects_by_type(object_type)) == 0:
                    continue

                # get the objectId
                objectId = random.choice(env.get_objects_by_type(object_type))['objectId']
                locations_from_which_object_is_visible: List[
                    Tuple[float, float, int, int]
                ] = []
                env.controller.step(
                    {
                        "action": "PositionsFromWhichItemIsInteractable",
                        "objectId": objectId,
                        "positions": positions_to_check_interactionable_from,
                        "horizon": 30,
                    }
                )
                assert (
                    env.last_action_success
                ), "Could not get positions from which item was interactable."                    
                returned = env.last_event.metadata["actionReturn"]  
                
                locations_from_which_object_is_visible.extend(
                    (
                        round(x, 2),
                        round(z, 2),
                        round_to_factor(rot, 90) % 360,
                        round_to_factor(hor, 30) % 360,
                    )
                    for x, z, rot, hor, standing in zip(
                        returned["x"],
                        returned["z"],
                        returned["rotation"],
                        returned["horizon"],
                        returned["standing"]
                    )
                    if standing == 1
                )
                if len(locations_from_which_object_is_visible) == 0:
                    continue

                while True:
                    # sample a location.
                    sampled_location = random.choice(locations_from_which_object_is_visible)

                    # teleport the agent to here.
                    env.teleport_agent_to(
                        x = sampled_location[0],
                        y = y,
                        z = sampled_location[1],
                        rotation = sampled_location[2],
                        horizon = sampled_location[3]
                    )
                
                    # check visibility
                    if env.last_action_success and env.get_object_by_id(objectId)['visible']:
                        break

                # get the infromation of bbox, segmentation mask.
                rgb_frame = env.current_frame
                instance_frame = env.current_instance_segmentation_frame
                visible_objects = [obj for obj in env.visible_objects() if obj['objectType'] in OBJECT_TYPES]
                N = (visible_objects)

                instance_detections2D = env.controller.last_event.instance_detections2D
                instance_masks = env.controller.last_event.instance_masks

                error_flag = False
                for obj in visible_objects:
                    if obj['objectId'] not in instance_masks or obj['objectId'] not in instance_detections2D:
                        error_flag = True
                
                if error_flag:
                    break

                boxes = [env.controller.last_event.instance_detections2D[obj['objectId']] for obj in visible_objects]
                boxes = np.stack(boxes, 0)
                labels = np.array([OBJECT_TYPES_TO_ID[obj['objectType']] for obj in visible_objects])
                area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0]) 
                iscrowd = False
                masks = [env.controller.last_event.instance_masks[obj['objectId']] for obj in visible_objects]
                masks = np.stack(masks, 0)

                # unique identifier for images.
                scene_name = floor_plan
                image_id = count

                # SAVE THE IMAGES. 
                image_path = os.path.join(args.save_path, 'images')
                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                file_name = 'ai2thor_%s_%09d.jpg' %(floor_plan, image_id)
                rgb_frame = PIL.Image.fromarray(rgb_frame)
                rgb_frame.save(os.path.join(image_path, file_name), 'JPEG')

                # save the instance segmentation
                segmentation_path = os.path.join(args.save_path, 'instance_segmentation')
                if not os.path.exists(segmentation_path):
                    os.makedirs(segmentation_path)

                instance_frame = PIL.Image.fromarray(instance_frame)
                instance_frame.save(os.path.join(segmentation_path, file_name), 'JPEG')



                to_save = {}
                to_save['boxes'] = boxes
                to_save['image_name'] = file_name
                to_save['labels'] = labels
                to_save['area'] = area
                to_save['iscrowd'] = iscrowd
                to_save['masks'] = masks
                to_save['scene_name'] = scene_name
                to_save['image_id'] = image_id

                anno_name = 'ai2thor_%s_%09d.pkl' %(floor_plan, image_id)
                # save the annotations.
                annotation_path = os.path.join(args.save_path, 'annotations')
                if not os.path.exists(annotation_path):
                    os.makedirs(annotation_path)
                
                with open(os.path.join(annotation_path, anno_name), 'wb') as f:
                    pickle.dump(to_save, f)

                count += 1

def parallel_main(args):
    procs = [mp.Process(target=main, args=(args, thread_num)) for thread_num in range(args.num_threads)]
    try:
        for proc in procs:
            proc.start()
            time.sleep(0.1)
    finally:
        for proc in procs:
            proc.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_threads", type=int, default=len(SCENE_NAMES), help="number of processes for parallel mode")
    parser.add_argument("--save_path", type=str, default='datasets/IThor/IThor_mask_rcnn_2', help="path_to_save the dataset.")
    parser.add_argument("--image_num", type=int, default=125, help="how many images sampled from each scene.")
    parser.add_argument("--shuffle_num", type=int, default=1, help="random shuffle scene after object.")
    
    args = parser.parse_args()
    gpu_ids = [0,1]

    args.gpu_ids = gpu_ids
    if args.num_threads > 1:
        parallel_main(args)
    else:
        main(args)