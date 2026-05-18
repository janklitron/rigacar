# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

import bpy
import mathutils
import math
import itertools
import re


# ---------------------------------------------------------------------------
# Cursor decorator
# ---------------------------------------------------------------------------

def cursor(cursor_mode):
    def cursor_decorator(func):
        def wrapper(self, context, *args, **kwargs):
            context.window.cursor_modal_set(cursor_mode)
            try:
                return func(self, context, *args, **kwargs)
            finally:
                context.window.cursor_modal_restore()
        return wrapper
    return cursor_decorator


# ---------------------------------------------------------------------------
# Bone name helpers
# ---------------------------------------------------------------------------

def bone_name(prefix, position, side, index=0):
    if index == 0:
        return '%s.%s.%s' % (prefix, position, side)
    else:
        return '%s.%s.%s.%03d' % (prefix, position, side, index)


def bone_range(bones, name_prefix, position, side):
    for index in itertools.count():
        name = bone_name(name_prefix, position, side, index)
        if name in bones:
            yield bones[name]
        else:
            break


def find_wheelbrake_bone(bones, position, side, index):
    other_side = 'R' if side == 'L' else 'L'
    name_prefix = 'WheelBrake'
    bone = bones.get(bone_name(name_prefix, position, side, index))
    if bone:
        return bone
    bone = bones.get(bone_name(name_prefix, position, other_side, index))
    if bone:
        return bone
    if index > 0:
        bone = bones.get(bone_name(name_prefix, position, side))
        if bone:
            return bone
        bone = bones.get(bone_name(name_prefix, position, other_side))
        if bone:
            return bone
    backward_compatible_bone_name = '%s Wheels' % ('Front' if position == 'Ft' else 'Back')
    return bones.get(backward_compatible_bone_name)


# ---------------------------------------------------------------------------
# FCurve helpers
# ---------------------------------------------------------------------------

def clear_property_animation(context, property_name, remove_keyframes=True):
    if remove_keyframes and context.object.animation_data and context.object.animation_data.action:
        fcurve_datapath = '["%s"]' % property_name
        action = context.object.animation_data.action
        fcurve = action.fcurves.find(fcurve_datapath)
        if fcurve is not None:
            action.fcurves.remove(fcurve)
    context.object[property_name] = .0


def create_property_animation(context, property_name):
    action = context.object.animation_data.action
    fcurve_datapath = '["%s"]' % property_name
    return action.fcurves.new(fcurve_datapath, index=0, action_group='Wheels rotation')


# ---------------------------------------------------------------------------
# Compatibility fix for old rigs
# ---------------------------------------------------------------------------

def fix_old_steering_rotation(rig_object):
    """Fix armature generated with rigacar version < 6.0"""
    if rig_object.pose and rig_object.pose.bones:
        if 'MCH-Steering.rotation' in rig_object.pose.bones:
            rig_object.pose.bones['MCH-Steering.rotation'].rotation_mode = 'QUATERNION'


# ---------------------------------------------------------------------------
# World-space pose sampler
#
# Replaces bpy.ops.nla.bake() which in Blender 4.x no longer creates an
# isolated temporary action — it overwrites the active action in place,
# destroying all existing keyframes.
#
# For each frame we move the scene clock, flush the depsgraph, then read
# pose_bone.matrix — the fully evaluated 4x4 world-space transform — for
# every requested bone. The active action is never touched.
# ---------------------------------------------------------------------------

class BoneSamples:
    """World-space matrix samples for a single bone, keyed by frame."""

    __slots__ = ('matrices',)

    def __init__(self):
        self.matrices = {}  # int -> mathutils.Matrix

    def record(self, frame, pose_bone):
        # pose_bone.matrix is the world-space 4x4 of the evaluated bone
        self.matrices[frame] = pose_bone.matrix.copy()

    def _get(self, frame):
        m = self.matrices.get(frame)
        if m is not None:
            return m
        keys = sorted(self.matrices)
        return self.matrices[keys[0]] if keys else mathutils.Matrix.Identity(4)

    def location(self, frame):
        return self._get(frame).to_translation()

    def rotation_quaternion(self, frame):
        return self._get(frame).to_quaternion()

    def scale(self, frame):
        return mathutils.Vector(self._get(frame).to_scale())


def sample_bones(context, frame_start, frame_end, bone_names):
    """
    Sample world-space pose matrices for bone_names over [frame_start, frame_end].
    Returns dict[str -> BoneSamples]. The active action is never modified.
    """
    scene = context.scene
    obj = context.object
    depsgraph = context.evaluated_depsgraph_get()
    original_frame = scene.frame_current

    result = {name: BoneSamples() for name in bone_names}

    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        depsgraph.update()
        obj_eval = obj.evaluated_get(depsgraph)
        for name, bone_samples in result.items():
            pb = obj_eval.pose.bones.get(name)
            if pb is not None:
                bone_samples.record(f, pb)

    scene.frame_set(original_frame)
    return result


# ---------------------------------------------------------------------------
# Base baking operator
# ---------------------------------------------------------------------------

class BakingOperator(object):
    frame_start: bpy.props.IntProperty(name='Start Frame', min=1)
    frame_end: bpy.props.IntProperty(name='End Frame', min=1)

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.data is not None
            and 'Car Rig' in context.object.data
            and context.object.data['Car Rig']
            and context.object.mode in ('POSE', 'OBJECT')
        )

    def invoke(self, context, event):
        if context.object.animation_data is None:
            context.object.animation_data_create()
        if context.object.animation_data.action is None:
            context.object.animation_data.action = bpy.data.actions.new(
                '%sAction' % context.object.name
            )
        action = context.object.animation_data.action
        self.frame_start = int(action.frame_range[0])
        self.frame_end = int(action.frame_range[1])
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        self.layout.prop(self, 'frame_start')
        self.layout.prop(self, 'frame_end')


# ---------------------------------------------------------------------------
# Wheel rotation bake operator
# ---------------------------------------------------------------------------

class ANIM_OT_carWheelsRotationBake(bpy.types.Operator, BakingOperator):
    bl_idname = 'anim.car_wheels_rotation_bake'
    bl_label = 'Bake wheels rotation'
    bl_description = 'Automatically generates wheels animation based on Root bone animation.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.object['wheels_on_y_axis'] = False
        self._bake_wheels_rotation(context)
        return {'FINISHED'}

    @cursor('WAIT')
    def _bake_wheels_rotation(self, context):
        bones = context.object.data.bones

        wheel_bones = []
        brake_bones = []
        for position, side in itertools.product(('Ft', 'Bk'), ('L', 'R')):
            for index, wheel_bone in enumerate(bone_range(bones, 'MCH-Wheel.rotation', position, side)):
                wheel_bones.append(wheel_bone)
                brake_bones.append(
                    find_wheelbrake_bone(bones, position, side, index) or wheel_bone
                )

        if not wheel_bones:
            return

        # Clear existing output FCurves before sampling so we don't corrupt data
        for wheel_bone in wheel_bones:
            clear_property_animation(context, wheel_bone.name.replace('MCH-', ''))

        # Sample world-space matrices for all relevant bones in one pass
        all_names = list({b.name for b in wheel_bones + brake_bones})
        samples = sample_bones(context, self.frame_start, self.frame_end, all_names)

        for wheel_bone, brake_bone in zip(wheel_bones, brake_bones):
            self._bake_single_wheel(
                context,
                samples[wheel_bone.name],
                samples[brake_bone.name],
                wheel_bone,
            )

    def _bake_single_wheel(self, context, wheel_samples, brake_samples, bone):
        """
        Compute cumulative roll angle for one wheel and write one keyframe per frame.

        Physics:
          - Each frame the wheel moves some world-space delta.
          - The component of that delta along the wheel's current forward axis
            gives signed speed (negative = reversing).
          - The brake bone's Y scale modulates speed:
              effective = raw * (2 * brake_scale_y - 1)
            At scale 0.5 the factor is 0 (stopped); 1.0 = full forward;
            0.0 = full reverse. This matches the original rig convention.
          - Dividing by radius converts linear distance to radians of rotation.
        """
        radius = bone.length if bone.length > 0.0 else 1.0
        # Rest-pose long axis of the bone in armature/world space
        bone_axis = (bone.head_local - bone.tail_local).normalized()

        fc = create_property_animation(context, bone.name.replace('MCH-', ''))

        distance = 0.0
        prev_loc = wheel_samples.location(self.frame_start)

        for f in range(self.frame_start, self.frame_end + 1):
            loc = wheel_samples.location(f)

            if f > self.frame_start:
                delta = loc - prev_loc

                # Brake modulation
                brake_y = brake_samples.scale(f).y
                delta = delta * (2.0 * brake_y - 1.0)

                # Signed speed along the bone's current world-space forward axis
                world_axis = wheel_samples.rotation_quaternion(f) @ bone_axis
                speed = math.copysign(delta.magnitude, world_axis.dot(delta))
                distance += speed / radius

            prev_loc = loc

            kf = fc.keyframe_points.insert(f, distance)
            kf.interpolation = 'LINEAR'
            kf.type = 'JITTER'


# ---------------------------------------------------------------------------
# Steering rotation bake operator
# ---------------------------------------------------------------------------

class ANIM_OT_carSteeringBake(bpy.types.Operator, BakingOperator):
    bl_idname = 'anim.car_steering_bake'
    bl_label = 'Bake car steering'
    bl_description = 'Automatically generates steering animation based on Root bone animation.'
    bl_options = {'REGISTER', 'UNDO'}

    rotation_factor: bpy.props.FloatProperty(name='Rotation factor', min=.1, default=1.0)

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        self.layout.prop(self, 'frame_start')
        self.layout.prop(self, 'frame_end')
        self.layout.prop(self, 'rotation_factor')

    def execute(self, context):
        if self.frame_end <= self.frame_start:
            return {'FINISHED'}

        arm_bones = context.object.data.bones
        if 'Steering' not in arm_bones or 'MCH-Steering.rotation' not in arm_bones:
            return {'FINISHED'}

        steering_bone = arm_bones['Steering']
        mch_bone = arm_bones['MCH-Steering.rotation']
        bone_offset = abs(steering_bone.head_local.y - mch_bone.head_local.y)

        self._bake_steering_rotation(context, bone_offset, mch_bone)
        return {'FINISHED'}

    @cursor('WAIT')
    def _bake_steering_rotation(self, context, bone_offset, mch_bone):
        """
        Derive steering angle from the MCH-Steering.rotation bone's movement
        and write one keyframe per frame into the 'Steering.rotation' property.

        The geometry: the MCH bone sits behind (or ahead of) the front axle by
        bone_offset units. As the car turns, the bone traces an arc. By projecting
        the per-frame travel vector onto the bone's forward/lateral axes we can
        recover the instantaneous turning radius, and from that the steering angle.
        """
        fix_old_steering_rotation(context.object)
        clear_property_animation(context, 'Steering.rotation')
        fc = create_property_animation(context, 'Steering.rotation')

        samples = sample_bones(
            context, self.frame_start, self.frame_end, [mch_bone.name]
        )
        bone_samples = samples[mch_bone.name]

        # Rest-pose axes of MCH-Steering.rotation in armature space
        bone_forward = (mch_bone.head_local - mch_bone.tail_local).normalized()
        bone_lateral = mathutils.Vector((1.0, 0.0, 0.0))

        prev_steering = 0.0

        for f in range(self.frame_start, self.frame_end + 1):
            rot = bone_samples.rotation_quaternion(f)
            world_forward = rot @ bone_forward
            world_lateral = rot @ bone_lateral

            if f < self.frame_end:
                travel = bone_samples.location(f + 1) - bone_samples.location(f)
            else:
                # Last frame: repeat previous value
                kf = fc.keyframe_points.insert(f, prev_steering)
                kf.type = 'JITTER'
                kf.interpolation = 'LINEAR'
                break

            forward_dist = travel.dot(world_forward)

            if forward_dist == 0.0:
                # Car is stationary or moving purely sideways — hold steering angle
                kf = fc.keyframe_points.insert(f, prev_steering)
                kf.type = 'JITTER'
                kf.interpolation = 'LINEAR'
                continue

            # Scale travel so its forward component equals bone_offset * rotation_factor,
            # then read off the lateral deviation as the steering displacement
            scaled_travel = travel * (bone_offset * self.rotation_factor / forward_dist)
            steering = mathutils.geometry.distance_point_to_plane(
                scaled_travel, world_forward, world_lateral
            )

            prev_steering = steering

            kf = fc.keyframe_points.insert(f, steering)
            kf.type = 'JITTER'
            kf.interpolation = 'LINEAR'


# ---------------------------------------------------------------------------
# Clear baked animation operator
# ---------------------------------------------------------------------------

class ANIM_OT_carClearSteeringWheelsRotation(bpy.types.Operator):
    bl_idname = 'anim.car_clear_steering_wheels_rotation'
    bl_label = 'Clear baked animation'
    bl_description = 'Clear generated rotation for steering and wheels'
    bl_options = {'REGISTER', 'UNDO'}

    clear_steering: bpy.props.BoolProperty(
        name='Steering',
        description='Clear generated animation for steering',
        default=True,
    )
    clear_wheels: bpy.props.BoolProperty(
        name='Wheels',
        description='Clear generated animation for wheels',
        default=True,
    )

    def draw(self, context):
        self.layout.use_property_decorate = False
        self.layout.label(text='Clear generated keyframes for')
        self.layout.prop(self, property='clear_steering')
        self.layout.prop(self, property='clear_wheels')

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.data is not None
            and context.object.data.get('Car Rig')
        )

    def execute(self, context):
        re_wheel = re.compile(r'^Wheel\.rotation\.(Ft|Bk)\.[LR](\.\d+)?$')
        for prop in context.object.keys():
            if prop == 'Steering.rotation':
                clear_property_animation(context, prop, remove_keyframes=self.clear_steering)
            elif re_wheel.match(prop):
                clear_property_animation(context, prop, remove_keyframes=self.clear_wheels)

        # Toggle object mode to force Blender to re-evaluate property drivers
        mode = context.object.mode
        bpy.ops.object.mode_set(mode='OBJECT' if mode == 'POSE' else 'POSE')
        bpy.ops.object.mode_set(mode=mode)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ANIM_OT_carWheelsRotationBake)
    bpy.utils.register_class(ANIM_OT_carSteeringBake)
    bpy.utils.register_class(ANIM_OT_carClearSteeringWheelsRotation)


def unregister():
    bpy.utils.unregister_class(ANIM_OT_carClearSteeringWheelsRotation)
    bpy.utils.unregister_class(ANIM_OT_carSteeringBake)
    bpy.utils.unregister_class(ANIM_OT_carWheelsRotationBake)


if __name__ == '__main__':
    register()
