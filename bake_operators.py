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


def clear_bone_location_animation(action, bone_name):
    """Remove all location FCurves for a pose bone from the action."""
    for index in range(3):
        datapath = 'pose.bones["%s"].location' % bone_name
        fc = action.fcurves.find(datapath, index=index)
        if fc is not None:
            action.fcurves.remove(fc)


def get_or_create_bone_location_fcurves(action, bone_name):
    """Return (fc_x, fc_y, fc_z) location FCurves for a pose bone, creating if missing."""
    datapath = 'pose.bones["%s"].location' % bone_name
    fcs = []
    for index in range(3):
        fc = action.fcurves.find(datapath, index=index)
        if fc is None:
            fc = action.fcurves.new(datapath, index=index, action_group='Suspension')
        fcs.append(fc)
    return fcs


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

        for wheel_bone in wheel_bones:
            clear_property_animation(context, wheel_bone.name.replace('MCH-', ''))

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
        radius = bone.length if bone.length > 0.0 else 1.0
        bone_axis = (bone.head_local - bone.tail_local).normalized()

        fc = create_property_animation(context, bone.name.replace('MCH-', ''))

        distance = 0.0
        prev_loc = wheel_samples.location(self.frame_start)

        for f in range(self.frame_start, self.frame_end + 1):
            loc = wheel_samples.location(f)

            if f > self.frame_start:
                delta = loc - prev_loc
                brake_y = brake_samples.scale(f).y
                delta = delta * (2.0 * brake_y - 1.0)
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
        fix_old_steering_rotation(context.object)
        clear_property_animation(context, 'Steering.rotation')
        fc = create_property_animation(context, 'Steering.rotation')

        samples = sample_bones(
            context, self.frame_start, self.frame_end, [mch_bone.name]
        )
        bone_samples = samples[mch_bone.name]

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
                kf = fc.keyframe_points.insert(f, prev_steering)
                kf.type = 'JITTER'
                kf.interpolation = 'LINEAR'
                break

            forward_dist = travel.dot(world_forward)

            if forward_dist == 0.0:
                kf = fc.keyframe_points.insert(f, prev_steering)
                kf.type = 'JITTER'
                kf.interpolation = 'LINEAR'
                continue

            scaled_travel = travel * (bone_offset * self.rotation_factor / forward_dist)
            steering = mathutils.geometry.distance_point_to_plane(
                scaled_travel, world_forward, world_lateral
            )

            prev_steering = steering

            kf = fc.keyframe_points.insert(f, steering)
            kf.type = 'JITTER'
            kf.interpolation = 'LINEAR'


# ---------------------------------------------------------------------------
# Suspension bake operator
#
# Drives a spring-damper simulation from the Root bone's per-frame
# acceleration, then writes the result as *local offset* keyframes on the
# Suspension bone — the same bone the animator moves by hand.
#
# ── Physics overview ────────────────────────────────────────────────────────
#
# 1. Root world-space velocity  v[i] = (pos[i+1] - pos[i-1]) / 2   (central
#    difference, jitter-resistant — avoids spikes from single-frame noise).
#
# 2. Acceleration  a[i] = (v[i+1] - v[i-1]) / 2   (same scheme).
#
# 3. The acceleration vector is expressed in the Root bone's *local* frame at
#    each frame, so the forces are always relative to the car's own heading —
#    braking always pitches the nose down regardless of world orientation.
#
# 4. Three independent scalar forces are extracted:
#      pitch_force  = -local_accel.y   (forward accel → nose lifts, minus = dip)
#      roll_force   =  local_accel.x   (rightward accel → body leans left)
#      heave_force  = -local_accel.z   (upward accel → body compresses down)
#
# 5. Each force feeds its own spring-damper ODE, integrated with a
#    semi-implicit (symplectic) Euler step scaled by dt = 1/fps so the
#    stiffness and damping sliders are fps-independent:
#
#      v_new = v + dt * (-k*x - c*v + force)
#      x_new = x + dt * v_new
#
#    Critical damping occurs at  c = 2 * sqrt(k).
#    Default k=4, c=4 → slightly under-damped, one gentle oscillation.
#
# 6. Output is clamped to ±max_offset (Blender units) so a runaway
#    integration never sends the bone flying off screen.
#
# 7. Keyframes are written as pure *local* offsets from the bone's rest
#    position (0, 0, 0 in pose space), matching exactly what the animator
#    sees when they grab the Suspension bone and move it by hand.
#    The rig's MCH-Body TRANSFORM constraints already map:
#      Suspension.loc.X → body roll
#      Suspension.loc.Y → body pitch
#      Suspension.loc.Z → body heave
# ---------------------------------------------------------------------------

class ANIM_OT_carSuspensionBake(bpy.types.Operator, BakingOperator):
    bl_idname = 'anim.car_suspension_bake'
    bl_label = 'Bake suspension'
    bl_description = (
        'Simulates inertia-based suspension: pitch on braking/acceleration, '
        'roll on cornering, heave on vertical movement. Writes local offset '
        'keyframes onto the Suspension bone.'
    )
    bl_options = {'REGISTER', 'UNDO'}

    # Spring stiffness k  (N/m equivalent — higher = snappier, less travel)
    stiffness: bpy.props.FloatProperty(
        name='Stiffness',
        description=(
            'Spring stiffness k. Higher values snap back faster and reduce '
            'travel. Critical damping at c = 2\u221ak. Typical range 1\u201320'
        ),
        min=0.1, max=100.0, default=4.0, step=10,
    )
    # Damping coefficient c  (critical at 2*sqrt(k))
    damping: bpy.props.FloatProperty(
        name='Damping',
        description=(
            'Damping coefficient c. At c = 2\u221ak the spring returns to rest '
            'without oscillating (critical damping). Lower = more bounce, '
            'higher = sluggish return'
        ),
        min=0.0, max=100.0, default=4.0, step=10,
    )

    # Per-axis influence multipliers — scale the *force input*, not the output
    pitch_factor: bpy.props.FloatProperty(
        name='Pitch',
        description='Strength of fore-aft (braking / acceleration) pitch response',
        min=0.0, max=5.0, default=1.0, step=10,
    )
    roll_factor: bpy.props.FloatProperty(
        name='Roll',
        description='Strength of lateral (cornering) roll response',
        min=0.0, max=5.0, default=1.0, step=10,
    )
    heave_factor: bpy.props.FloatProperty(
        name='Heave',
        description='Strength of vertical (bump / crest) heave response',
        min=0.0, max=5.0, default=0.5, step=10,
    )

    # Output clamp — keeps the bone from flying away if the rig is huge
    max_offset: bpy.props.FloatProperty(
        name='Max offset',
        description='Maximum local displacement (Blender units) in any axis',
        min=0.001, max=10.0, default=0.5, step=1,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, 'frame_start')
        layout.prop(self, 'frame_end')
        layout.separator()

        layout.label(text='Spring')
        layout.prop(self, 'stiffness')
        layout.prop(self, 'damping')
        layout.separator()

        layout.label(text='Axis influence')
        layout.prop(self, 'pitch_factor')
        layout.prop(self, 'roll_factor')
        layout.prop(self, 'heave_factor')
        layout.separator()

        layout.prop(self, 'max_offset')

    @classmethod
    def poll(cls, context):
        if not super().poll(context):
            return False
        return (
            context.object.data.bones.get('Suspension') is not None
            and context.object.data.bones.get('Root') is not None
        )

    def execute(self, context):
        if self.frame_end <= self.frame_start:
            return {'FINISHED'}
        self._bake_suspension(context)
        return {'FINISHED'}

    @cursor('WAIT')
    def _bake_suspension(self, context):
        obj = context.object
        action = obj.animation_data.action
        fps = context.scene.render.fps / context.scene.render.fps_base
        dt = 1.0 / fps

        # ── 1. Sample Root bone world-space positions ────────────────────────
        frames = list(range(self.frame_start, self.frame_end + 1))
        n = len(frames)

        samples = sample_bones(context, self.frame_start, self.frame_end, ['Root'])
        root_s = samples['Root']

        locs = [root_s.location(f) for f in frames]
        rots = [root_s.rotation_quaternion(f) for f in frames]

        # ── 2. Central-difference velocity & acceleration ────────────────────
        # Central difference: v[i] = (p[i+1] - p[i-1]) / (2*dt)
        # Clamped at boundaries with forward/backward difference.
        vels = []
        for i in range(n):
            if i == 0:
                v = (locs[1] - locs[0]) / dt if n > 1 else mathutils.Vector()
            elif i == n - 1:
                v = (locs[-1] - locs[-2]) / dt
            else:
                v = (locs[i + 1] - locs[i - 1]) / (2.0 * dt)
            vels.append(v)

        accels = []
        for i in range(n):
            if i == 0:
                a = (vels[1] - vels[0]) / dt if n > 1 else mathutils.Vector()
            elif i == n - 1:
                a = (vels[-1] - vels[-2]) / dt
            else:
                a = (vels[i + 1] - vels[i - 1]) / (2.0 * dt)
            accels.append(a)

        # ── 3. Express acceleration in Root bone local space ─────────────────
        # This makes the forces car-heading-relative so braking always pitches
        # the nose down regardless of which direction the car is facing.
        # Root bone local axes: Y = forward, X = right, Z = up.
        local_accels = []
        for i in range(n):
            # Inverse of the bone's world rotation maps world→local
            world_to_local = rots[i].inverted()
            local_accels.append(world_to_local @ accels[i])

        # Extract scalar forces from local acceleration components.
        # Sign conventions match the rig's MCH-Body TRANSFORM constraints:
        #   braking  → negative local Y accel → pitch nose down  (positive pitch_force)
        #   cornering right → positive local X accel → lean left (positive roll_force)
        #   going over bump → positive local Z accel → compress  (positive heave_force)
        pitch_forces = [-a.y * self.pitch_factor for a in local_accels]
        roll_forces  = [ a.x * self.roll_factor  for a in local_accels]
        heave_forces = [-a.z * self.heave_factor for a in local_accels]

        # ── 4. Spring-damper integration (semi-implicit Euler, fps-scaled) ───
        #
        # ODE:  x'' = -k*x - c*x' + force(t)
        # Semi-implicit Euler (symplectic, unconditionally stable for c >= 0):
        #   v_new = v + dt * (-k*x - c*v + force)
        #   x_new = x + dt * v_new
        #
        # When the clamp is hit, zero velocity so stored energy is not released
        # as an uncontrolled rebound once forcing stops (travel-limit collision).
        # Everything else — including the spring-back to rest when the car stops
        # — is handled naturally by the ODE: force drops to zero, the spring
        # term -k*x pulls x back, damping bleeds off the oscillation.
        k = self.stiffness
        c = self.damping
        clamp = self.max_offset

        def integrate(forces):
            x, v = 0.0, 0.0
            result = []
            for force in forces:
                # Semi-implicit Euler step
                v = v + dt * (-k * x - c * v + force)
                x = x + dt * v

                # Travel limit: absorb kinetic energy at the wall
                if x > clamp:
                    x = clamp
                    v = 0.0
                elif x < -clamp:
                    x = -clamp
                    v = 0.0

                result.append(x)
            return result

        disp_pitch = integrate(pitch_forces)
        disp_roll  = integrate(roll_forces)
        disp_heave = integrate(heave_forces)

        # ── 5. Write keyframes as pure local bone offsets ────────────────────
        # pose_bone.location is the delta from the bone's rest position in
        # pose space — exactly (0,0,0) when the bone hasn't been moved.
        # Writing small displacement values here is identical to the animator
        # grabbing the Suspension bone and nudging it, which is exactly what
        # the MCH-Body constraints are designed to read.
        clear_bone_location_animation(action, 'Suspension')
        fc_x, fc_y, fc_z = get_or_create_bone_location_fcurves(action, 'Suspension')

        for i, f in enumerate(frames):
            kf = fc_x.keyframe_points.insert(f, disp_roll[i])
            kf.interpolation = 'LINEAR'
            kf.type = 'JITTER'

            kf = fc_y.keyframe_points.insert(f, disp_pitch[i])
            kf.interpolation = 'LINEAR'
            kf.type = 'JITTER'

            kf = fc_z.keyframe_points.insert(f, disp_heave[i])
            kf.interpolation = 'LINEAR'
            kf.type = 'JITTER'


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
    clear_suspension: bpy.props.BoolProperty(
        name='Suspension',
        description='Clear generated suspension animation from the Suspension bone',
        default=True,
    )

    def draw(self, context):
        self.layout.use_property_decorate = False
        self.layout.label(text='Clear generated keyframes for')
        self.layout.prop(self, property='clear_steering')
        self.layout.prop(self, property='clear_wheels')
        self.layout.prop(self, property='clear_suspension')

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

        if self.clear_suspension:
            anim = context.object.animation_data
            if anim and anim.action:
                clear_bone_location_animation(anim.action, 'Suspension')

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
    bpy.utils.register_class(ANIM_OT_carSuspensionBake)
    bpy.utils.register_class(ANIM_OT_carClearSteeringWheelsRotation)


def unregister():
    bpy.utils.unregister_class(ANIM_OT_carClearSteeringWheelsRotation)
    bpy.utils.unregister_class(ANIM_OT_carSuspensionBake)
    bpy.utils.unregister_class(ANIM_OT_carSteeringBake)
    bpy.utils.unregister_class(ANIM_OT_carWheelsRotationBake)


if __name__ == '__main__':
    register()
