# Jupyter kernel: tf2
import os
import imageio

# LLFF poses library for COLMAP
import collections
import numpy as np
import struct


# Code for extracting the poses into an npy file from COLMAP's output.

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):

    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """

    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Poses:

    # TODO: remove redundancy parameters
    COLMAP_MODEL = 0
    SCALE_PARAM = 1.0

    def __init__(self, rootdir, imgdir):
        self.rootdir = rootdir
        self.imgsdir = imgdir

    def run(self):
        points, pts3d, perm, poses = self.extract_poses()
        save_arr = self.save_poses(self.rootdir, poses, pts3d, perm)
        #bds = self.fetch_min_max_bds(save_arr)
        imgs, focal, poses_np, bds = self.rescale(8)

        # Scene bounds
        near = float(bds.min()) * 1.0
        far = float(bds.max()) * 0.9
        print("Scene bounds:", near, far)
        np.savez_compressed(f'{self.rootdir}/data', images=imgs, poses=poses_np, focal=focal, bds=bds)


    def extract_poses(self):

        points3dfile = os.path.join(self.rootdir, f'sparse/{self.COLMAP_MODEL}/points3D.bin')
        pts3d = read_points3d_binary(points3dfile)

        print("Get world scaling")
        points = np.stack([p.xyz for p in pts3d.values()])
        cen = np.median(points, axis=0)
        points -= cen
        dists = (points ** 2).sum(axis=1)

        # FIXME: Questionable autoscaling. Adopt method from Noah Snavely
        # meddist = np.median(dists)
        # points *= 2 * SCALE_PARAM / meddist
        #

        camerasfile = os.path.join(self.rootdir, f'sparse/{self.COLMAP_MODEL}/cameras.bin')
        camdata = read_cameras_binary(camerasfile)

        list_of_keys = list(camdata.keys())
        cam = camdata[list_of_keys[0]]
        print('Cameras keys:', len(cam))

        h, w, f = cam.height, cam.width, cam.params[0]
        hwf = np.array([h, w, f]).reshape([3, 1])

        imagesfile = os.path.join(self.rootdir, f'sparse/{self.COLMAP_MODEL}/images.bin')
        imdata = read_images_binary(imagesfile)

        w2c_mats = []

        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

        names = [imdata[k].name for k in imdata]
        print('Images #', len(names))
        perm = np.argsort(names)

        #
        normal_cameras = np.empty(shape=(0, 3, 1))
        c2w_mats = []
        #

        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape([3, 1])

            # w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            # w2c_mats.append(w2c)

            #
            t_world = -R.T @ t
            t_world = (t_world - cen[:, None])
            # t_world = (t_world - cen[:, None])* 2 * SCALE_PARAM / meddist
            normal_cameras = np.vstack((normal_cameras, np.expand_dims(t_world, axis=0)))
            c2w = np.concatenate([np.concatenate([R.T, t_world], 1), bottom], 0)
            c2w_mats.append(c2w)
            #

        # w2c_mats = np.stack(w2c_mats, 0)
        # c2w_mats = np.linalg.inv(w2c_mats)

        #
        c2w_mats = np.stack(c2w_mats, 0)
        #

        poses = c2w_mats[:, :3, :4]
        poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[0]]).transpose([2, 0, 1])], 2)

        return points, pts3d, perm, poses

    @staticmethod
    def save_poses(basedir, poses, pts3d, perm):
        pts_arr = []
        vis_arr = []
        for k in pts3d:
            pts_arr.append(pts3d[k].xyz)
            cams = [0] * poses.shape[0]
            for ind in pts3d[k].image_ids:
                # print(len(cams))
                # print(ind)
                if len(cams) < ind - 1:
                    print('ERROR: the correct camera poses for current points cannot be accessed')
                    return
                cams[ind - 1] = 1
            vis_arr.append(cams)

        pts_arr = np.array(pts_arr)
        vis_arr = np.array(vis_arr)
        print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

        # ray origin + ray direction ! sum over spatial dims, inner product poses[:,:3, 2:3] is a unit vector.
        # # so it is the projection's length on the z axis (camera direction).
        zvals = np.sum((pts_arr[:, :, np.newaxis].transpose([2, 1, 0]) - poses[:, :3, 3:4]) * poses[:, :3, 2:3], 1)

        zvals = zvals.transpose([1, 0])
        valid_z = zvals[vis_arr == 1]
        # print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )

        save_arr = []
        for i in perm:
            vis = vis_arr[:, i]
            zs = zvals[:, i]
            zs = zs[vis == 1]
            close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
            print(i, close_depth, inf_depth)

            save_arr.append(np.concatenate([poses[i, ...].ravel(), np.array([close_depth, inf_depth])], 0))
        save_arr = np.array(save_arr)
        print(save_arr.shape)
        np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

        # save_arr 3x5+2 array pose values + 2 the boundaries values.
        return save_arr

    def fetch_min_max_bds(self, save_arr):
        bds = save_arr[:, -2:]
        print('Loaded', self.rootdir, bds.min(), bds.max())
        bds = bds.astype(np.float32)
        return bds

    def rescale(self, factor):
        def minify(basedir, factor):

            from subprocess import check_output

            # imgdir = os.path.join(basedir, 'images')
            imgs = [os.path.join(self.imgsdir, f) for f in sorted(os.listdir(self.imgsdir))]
            imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
            imgdir_orig = self.imgsdir

            wd = os.getcwd()

            name = 'images_{}'.format(factor)
            resizearg = '{}%'.format(int(100. / factor))
            imgdir = os.path.join(basedir, name)

            try:
                os.makedirs(imgdir)
                check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
            except FileExistsError:
                return

            print('Minifying', factor, basedir)

            ext = imgs[0].split('.')[-1]
            args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
            print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)

            if ext != 'png':
                check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
                print('Removed duplicates')
            print('Done')

        poses_arr = np.load(os.path.join(self.rootdir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        bds = poses_arr[:, -2:]

        sfx = '_{}'.format(factor)
        minify(self.rootdir, 8)

        imgdir = os.path.join(self.rootdir, 'images' + sfx)

        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                    f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        if poses.shape[0] != len(imgfiles):
            print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[0]))

        sh = imageio.v2.imread(imgfiles[0]).shape
        poses[:, :2, 4:] = np.array(sh[:2]).reshape([2, 1])
        poses[:, 2, 4] = poses[:, 2, 4] * 1. / factor

        def imread(f):
            if f.endswith('png'):
                return imageio.v2.imread(f, ignoregamma=True)
            else:
                return imageio.v2.imread(f)

        imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
        imgs = np.stack(imgs, 0)

        focal = np.array(poses[0, -1, -1])

        print('Loaded image data', imgs.shape, poses.shape, poses[0, :, -1])

        def add_bottum_axis(poses):

            bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
            poses = poses[:, :, :-1]
            poses_np = np.empty((0, 4, 4))

            for i in range(poses.shape[0]):
                poses_np = np.vstack((poses_np, np.concatenate([poses[i], bottom], 0).reshape(1, 4, 4)))
            return poses_np

        poses_np = add_bottum_axis(poses)

        # summary
        print("poses:", poses_np.shape)
        print("images:", imgs.shape)
        print("focal:", focal)
        print("bds:", bds.shape)

        return imgs, focal, poses_np, bds
