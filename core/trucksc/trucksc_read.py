
from truckscenes import TruckScenes
from truckscenes.utils.splits import create_splits_scenes



def all_scene_tokens(trucksc: TruckScenes) -> list[str]:
    """Returns all scene tokens in the dataset."""
    scene_tokens_all = [s['token'] for s in trucksc.scene]
    if len(scene_tokens_all) == 0:
        raise RuntimeError("Error: Database has no samples!")
    
    # Remove scene_token in index 345 because it's missing a camera token
    if trucksc.version == 'v1.0-trainval':
        scene_tokens_all.pop(345)
    
    return scene_tokens_all


def illuminated_scene_tokens(trucksc: TruckScenes) -> list[str]:
    """Returns the scene tokens for the illuminated scenes."""
    scene_tokens = []
    for scene_token in all_scene_tokens(trucksc):
        scene_record = trucksc.get('scene', scene_token)
        if 'illuminated' in scene_record['description']:
            scene_tokens.append(scene_token)
    return scene_tokens


def split_scene_tokens(trucksc: TruckScenes, eval_split: str, illuminated_scenes_only: bool = False) -> list[str]:
    """Returns the scene tokens for the given split."""
    splits = create_splits_scenes()
    if illuminated_scenes_only:
        orig_scene_tokens = illuminated_scene_tokens(trucksc)
    else:
        orig_scene_tokens = all_scene_tokens(trucksc)
    scene_tokens = []
    for scene_token in orig_scene_tokens:
        scene_record = trucksc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
                scene_tokens.append(scene_token)
    return scene_tokens

def next_sample_available(trucksc: TruckScenes, sample_data_tokens_list: list[str]) -> bool:
    """
    Check if the next sample is available for all the cameras.
    """
    for sample_data_token in sample_data_tokens_list:
        sample_data = trucksc.get('sample_data', sample_data_token)
        if sample_data['next'] == '':
            return False
    return True


def get_scene_samples_data(trucksc: TruckScenes, scene_token: str) -> list[dict]:
    """
    Return all samples for a given scene.

    The returned samples are different from the ones in the truckscenes library.
    Here, all the camera frames are collected, and then the radar and lidar measurements are collected for each camera frame.

    Each dictionary in the the list contains data the same as a sample in the truckscenes library.

    Args:
        trucksc (TruckScenes): The truckscenes object.
        scene_token (str): The token of the scene.

    Returns:
        list[dict]: A list of dictionaries, each containing the camera, radar, and lidar tokens for a given timestamp.
    """


    # Start by getting the first camera images
    scene_rec = trucksc.get('scene', scene_token)
    first_sample_token = scene_rec['first_sample_token']
    frist_sample = trucksc.get('sample', first_sample_token)
    curr_cameras_sample_data_tokens = [token for key, token in frist_sample['data'].items() if 'CAMERA' in key]

    sensor_types = [sensor_type for sensor_type in frist_sample['data'].keys() if 'CAMERA' not in sensor_type]

    # According to the camera freq, which is lower than the radar freq
    freq = 10
    time_step = 1 / freq * 1e6
    # We will use this time to find the radar and lidar closest to the current time
    curr_time = frist_sample['timestamp']

    sample_data_dict = frist_sample['data'].copy()
    samples_data_list = [sample_data_dict]

    while next_sample_available(trucksc, curr_cameras_sample_data_tokens):
        curr_cameras_sample_data_tokens = [trucksc.get('sample_data', token)['next'] for token in curr_cameras_sample_data_tokens]
        curr_time += time_step

        # For cameras, only increment the token
        sample_data_dict = {key: trucksc.get('sample_data', token)['next'] for key, token in sample_data_dict.items() if 'CAMERA' in key}

        # For each radar and lidar, find the closest sample data token to curr_time
        for sensor_type in sensor_types:
            initial_sensor_sample_data_token = frist_sample['data'][sensor_type]
            curr_sensor_sample_data_token = initial_sensor_sample_data_token
            curr_sensor_sample_data = trucksc.get('sample_data', curr_sensor_sample_data_token)
            while curr_sensor_sample_data['timestamp'] < curr_time:
                curr_sensor_sample_data_token = curr_sensor_sample_data['next']
                curr_sensor_sample_data = trucksc.get('sample_data', curr_sensor_sample_data_token)
            # curr_sensor_sample_data is in time larger than curr_time. Check which one is closer, this or the previous one
            prev_sensor_sample_data = trucksc.get('sample_data', curr_sensor_sample_data['prev'])
            if abs(curr_time - curr_sensor_sample_data['timestamp']) < abs(curr_time - prev_sensor_sample_data['timestamp']):
                sample_data_dict[sensor_type] = curr_sensor_sample_data['token']
            else:
                sample_data_dict[sensor_type] = prev_sensor_sample_data['token']
        
        samples_data_list.append(sample_data_dict)

    return samples_data_list

        
        
    


