import os.path as osp
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import PointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import transform_matrix

from core.trucksc import frame_conversion


def get_radial_velocity(fused_radar_pointcloud: RadarPointCloud) -> np.ndarray:
    """
    Returns the radial velocity of each point in the fused radar pointcloud.
    """
    pcl = fused_radar_pointcloud.points.copy()
    # Calculate magnitude of velocity
    velocity_magnitude = np.linalg.norm(pcl[3:6, :], axis=0)
    # Velocity sign (Direction of radial velocity)
    velocity_sign = np.sign(np.sum(pcl[3:6, :] * pcl[:3, :], axis=0))
    # Radial velocity
    radial_velocity = velocity_magnitude * velocity_sign
    return radial_velocity




def from_file_multisweep(
        cls,
        trucksc,
        sample_rec: Dict,
        chan: str,
        ref_sd_token: str,
        nsweeps: int = 1,
        min_distance: float = 0.0
    ) -> Tuple[PointCloud, np.ndarray]:
    """
    Return a point cloud that aggregates multiple sweeps.
    As every sweep is in a different coordinate frame, we need to map the coordinates
    to a single reference frame.
    As every sweep has a different timestamp, we need to account for that in
    the transformations and timestamps.

    Arguments:
        trucksc: A TruckScenes instance.
        sample_rec: The current sample.
                    sample should include: 
                    - data.ref_chan: token of the reference channel
                    - data.chan: token of the channel to aggregate
        chan: The lidar/radar channel from which we track back n sweeps to aggregate
            the point cloud.
        ref_chan: The reference channel of the current sample_rec that the
            point clouds are mapped to.
        nsweeps: Number of sweeps to aggregated.
        min_distance: Distance below which points are discarded.

    Returns:
        all_pc: The aggregated point cloud.
        all_times: The aggregated timestamps.
    """
    # Init.
    points = np.zeros((cls.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    all_pc = cls(points, timestamps)
    all_times = np.zeros((1, 0))

    all_pc.points = np.zeros((cls.nbr_dims()+1, 0), dtype=np.float64)

    # Get reference pose and timestamp.
    ref_sd_rec = trucksc.get('sample_data', ref_sd_token)
    ref_pose_rec = trucksc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = trucksc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                    Quaternion(ref_cs_rec['rotation']),
                                    inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'],
                                        Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data'][chan]
    current_sd_rec = trucksc.get('sample_data', sample_data_token)

    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = cls.from_file(osp.join(trucksc.dataroot, current_sd_rec['filename']))
        radial_velocity = get_radial_velocity(current_pc).astype(np.float64)
        current_pc.points = np.concatenate((current_pc.points, radial_velocity[None, :]), axis=0)
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = trucksc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']),
                                            inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = trucksc.get('calibrated_sensor',
                                        current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'],
                                            Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global,
                                        global_from_car, car_from_current])
        # current_pc.transform(trans_matrix)
        current_pc = frame_conversion.r_pc_transform(current_pc, trans_matrix)

        # Add time vector which can be used as a temporal feature.
        if current_pc.timestamps is not None:
            # Per point difference
            time_lag = ref_time - 1e-6 * current_pc.timestamps
        else:
            # Difference to sample data
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            time_lag = time_lag * np.ones((1, current_pc.nbr_points()))
        all_times = np.hstack((all_times, time_lag))

        # Merge with key pc.
        all_pc.points = np.hstack((all_pc.points, current_pc.points))
        if current_pc.timestamps is not None:
            all_pc.timestamps = np.hstack((all_pc.timestamps, current_pc.timestamps))

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = trucksc.get('sample_data', current_sd_rec['prev'])

    return all_pc, all_times


def get_fused_radar_pointcloud(
        trucksc: TruckScenes,
        sample: Dict[str, Any],
        ref_sd_token: Optional[str] = None,
        nsweeps: int = 1
    ) -> RadarPointCloud:
    """ Returns a fused radar point cloud for the given sample.

    Fuses the point clouds of the given sample and returns them in the reference
    sensor frame at the timestamp of the reference sample data record. Uses the
    timestamp of the sample data record of the individual sensors to transform
    them to a uniform frame.

    Does not consider the timestamps of the individual points during the
    fusion. Therefore, motion distortion is not considered and deskewing
    is not performed.

    Arguments:
        trucksc: TruckScenes dataset instance.
        sample: Reference sample to fuse the point clouds of.

    Returns:
        fused_point_cloud: Fused radar point cloud in the ego vehicle frame at the
            timestamp of the sample.
    """
    # Initialize
    points = np.zeros((RadarPointCloud.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    fused_point_cloud = RadarPointCloud(points, timestamps)
    fused_point_cloud.points = np.zeros((RadarPointCloud.nbr_dims()+1, 0), dtype=np.float64)

    # Define reference sensor
    if ref_sd_token is None:
        ref_sd_token = sample['data']['LIDAR_LEFT']

    # Iterate over all radar sensors and fuse their point clouds
    for sensor in sample['data'].keys():
        if 'radar' not in sensor.lower():
            continue

        # Load pointcloud
        point_cloud, _ = from_file_multisweep(RadarPointCloud, trucksc, sample, chan=sensor, ref_sd_token=ref_sd_token, nsweeps=nsweeps)
        point_cloud = frame_conversion.transform_r_pcl_from_ref_to_vehicle_flat_up(trucksc, point_cloud, ref_sd_token)

        # Merge with reference point cloud.
        fused_point_cloud.points = np.hstack((fused_point_cloud.points, point_cloud.points))
        if point_cloud.timestamps is not None:
            fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, point_cloud.timestamps))

    # # Calculate radial velocity at the current frame, on the x-y plane
    # # To do that, set z and Vz to zero for the calculation
    # tmp_fused_point_cloud = deepcopy(fused_point_cloud)
    # tmp_fused_point_cloud.points[2, :] = 0.0
    # tmp_fused_point_cloud.points[5, :] = 0.0
    # radial_velocity_direct = get_radial_velocity_direct(tmp_fused_point_cloud)
    # radial_velocity = get_radial_velocity(tmp_fused_point_cloud)
    # # print(f'radial_velocity_direct: {radial_velocity_direct[:5]}')
    # # print(f'radial_velocity: {radial_velocity[:5]}')
    # assert not np.any(np.sign(radial_velocity) != np.sign(radial_velocity_direct))

    # fused_point_cloud.points = np.concatenate((fused_point_cloud.points, radial_velocity[None, :]), axis=0)
    return fused_point_cloud





########################################################
# Filtering functions
########################################################


def get_fused_radar_pointcloud_in_ref_frame(
        trucksc: TruckScenes,
        sample: Dict[str, Any],
        ref_sd_token: Optional[str] = None,
        nsweeps: int = 1
    ) -> RadarPointCloud:
    # Initialize
    points = np.zeros((RadarPointCloud.nbr_dims(), 0), dtype=np.float64)
    timestamps = np.zeros((1, 0), dtype=np.uint64)
    fused_point_cloud = RadarPointCloud(points, timestamps)
    fused_point_cloud.points = np.zeros((RadarPointCloud.nbr_dims()+1, 0), dtype=np.float64)

    # Define reference sensor
    if ref_sd_token is None:
        ref_sd_token = sample['data']['LIDAR_LEFT']

    # Iterate over all radar sensors and fuse their point clouds
    for sensor in sample['data'].keys():
        if 'radar' not in sensor.lower():
            continue

        # Load pointcloud
        point_cloud, _ = from_file_multisweep(RadarPointCloud, trucksc, sample, chan=sensor, ref_sd_token=ref_sd_token, nsweeps=nsweeps)

        # Merge with reference point cloud.
        fused_point_cloud.points = np.hstack((fused_point_cloud.points, point_cloud.points))
        if point_cloud.timestamps is not None:
            fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, point_cloud.timestamps))

    return fused_point_cloud


def ref_frame_pcl_to_camera_view_filter_mask(
        trucksc: TruckScenes,
        ref_frame_pcl: RadarPointCloud,
        ref_sd_token: str,
        camera_token: str,
        min_dist: float = 0.0
    ) -> np.ndarray:
    ref_frame_pcl = deepcopy(ref_frame_pcl)

    cam = trucksc.get('sample_data', camera_token)
    im = Image.open(osp.join(trucksc.dataroot, cam['filename']))
    
    _, _,transform_matrix = frame_conversion.sensor_from_sensor(trucksc, ref_sd_token, camera_token)
    camera_frame_pcl = frame_conversion.r_pc_transform(ref_frame_pcl, transform_matrix)
    
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    cs_record = trucksc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points = frame_conversion.view_points(camera_frame_pcl.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    depths = camera_frame_pcl.points[2, :]

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to
    # avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    return mask


def get_points_in_view_mask(
        trucksc: TruckScenes,
        sample: Dict[str, Any],
        ref_sd_token: str,
        camera_tokens: List[str],
        nsweeps: int = 1,
    ) -> np.ndarray:
    """
    Returns a mask of the points that are in the camera views.

    Arguments:
        fused_radar_pointcloud: Fused radar pointcloud.
        trucksc: TruckScenes dataset instance.
        camera_token: Camera sample_data token.
    """
    fused_radar_pcl_in_ref_frame = get_fused_radar_pointcloud_in_ref_frame(trucksc, sample, ref_sd_token, nsweeps)
    mask = np.zeros(fused_radar_pcl_in_ref_frame.points.shape[1], dtype=bool)
    for camera_token in camera_tokens:
        filter_mask = ref_frame_pcl_to_camera_view_filter_mask(
            trucksc,
            fused_radar_pcl_in_ref_frame,
            ref_sd_token,
            camera_token)
        mask = np.logical_or(mask, filter_mask)
    return mask


def filter_points_in_same_pixel_location(fused_radar_pointcloud: RadarPointCloud, image_size: int) -> np.ndarray:
    """
    Filters out points that are in the same pixel location. Take the one with the highest RCS value.
    """
    mask = np.zeros(fused_radar_pointcloud.points.shape[1], dtype=bool)
    locations = fused_radar_pointcloud.points[:3, :]
    rcs_values = fused_radar_pointcloud.points[6, :]
    
    # Convert x,y coordinates to integer pixel coordinates
    pixel_coords = {}  # Dictionary to group points by pixel location
    
    for i in range(locations.shape[1]):
        x, y = int(locations[0, i]), int(locations[1, i])
        pixel_key = (x, y)
        
        if pixel_key not in pixel_coords:
            pixel_coords[pixel_key] = []
        pixel_coords[pixel_key].append(i)
    
    # For each pixel location, keep only the point with highest RCS value
    for indices in pixel_coords.values():
        if len(indices) == 1:
            # Only one point at this pixel location
            mask[indices[0]] = True
        else:
            # Multiple points at same pixel location - keep the one with highest RCS
            rcs_at_pixel = [rcs_values[i] for i in indices]
            max_rcs_idx = np.argmax(rcs_at_pixel)
            mask[indices[max_rcs_idx]] = True
    
    return mask


def filter_fused_radar_pointcloud(
        fused_radar_pointcloud: RadarPointCloud,
        trucksc: TruckScenes,
        sample: Dict[str, Any],
        ref_sd_token: Optional[str] = None,
        image_size: int = 512,
        point_limit: float = 50.0,
        camera_tokens: Optional[List[str]] = None,
        nsweeps: int = 1,
        set_z: Optional[float] = None
    ) -> RadarPointCloud:
    """
    Filters a fused radar pointcloud to a given range and camera views.

    What's filtered:
    - Points outside the range of the point_limit
    - Points outside the camera views
    - Points too close the the cameras (min_dist)
    - Points that are too far from the x-y plane
    - Multiple points at the same location in the x-y plane (After rounding with int)
    """
    fused_radar_pointcloud = deepcopy(fused_radar_pointcloud)
    if ref_sd_token is None:
        ref_sd_token = sample['data']['LIDAR_LEFT']

    # Filter out points that are not in the camera views
    if camera_tokens is not None:
        camera_views_mask = get_points_in_view_mask(trucksc, sample, ref_sd_token, camera_tokens, nsweeps)
        fused_radar_pointcloud.points = fused_radar_pointcloud.points[:, camera_views_mask]

    # Filter out points in a location larger or lower than point_limit.
    # This will later keep the current middle of image which represents the ego vehicle
    locations = fused_radar_pointcloud.points[:3, :]
    range_limit_mask = np.logical_and(np.all(locations < point_limit, axis=0), 
                                      np.all(locations > -point_limit, axis=0))
    fused_radar_pointcloud.points = fused_radar_pointcloud.points[:, range_limit_mask]

    if set_z is not None:
        fused_radar_pointcloud.points[2, :] = set_z
    
    # Reshape locations to fit the image size
    # locations = fused_radar_pointcloud.points[:3, :]
    # locations = (locations - (-point_limit)) / (2*point_limit)
    # # Convert locations to [0, 512]. We will have 512 bins for each axis. with indices [0, 511]
    # locations = locations * image_size  # [0,1] -> [0,512]
    # fused_radar_pointcloud.points[:3, :] = locations

    # Filter out points that are in the same pixel location, take the one with the highest RCS value
    # same_location_mask = filter_points_in_same_pixel_location(fused_radar_pointcloud, image_size)
    # fused_radar_pointcloud.points = fused_radar_pointcloud.points[:, same_location_mask]

    return fused_radar_pointcloud