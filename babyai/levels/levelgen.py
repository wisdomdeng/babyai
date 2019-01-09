import random
from collections import OrderedDict
from copy import deepcopy
import gym
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.minigrid import LockedDoor
from .verifier import *
import numpy as np
import pickle
from gym_minigrid.minigrid import Key, Ball, Box
from gym_minigrid.minigrid import MiniGridEnv

class RejectSampling(Exception):
    """
    Exception used for rejection sampling
    """

    pass


class RoomGridLevel(RoomGrid):
    """
    Base for levels based on RoomGrid
    A level, given a random seed, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """

    def __init__(
        self,
        room_size=8,
        **kwargs
    ):
        super().__init__(
            room_size=room_size,
            **kwargs
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.instrs.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = self.room_size ** 2
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        num_navs = self.num_navs_needed(self.instrs)
        self.max_steps = num_navs * nav_time_maze

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we drop an object, we need to update its position in the environment
        if action == self.actions.drop:
            self.update_objs_poss()

        # If we've successfully completed the mission
        status = self.instrs.verify(action)

        if status is 'success':
            done = True
            reward = self._reward()
        elif status is 'failure':
            done = True
            reward = 0

        return obs, reward, done, info

    def update_objs_poss(self, instr=None):
        if instr is None:
            instr = self.instrs
        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr) or isinstance(instr, AfterInstr):
            self.update_objs_poss(instr.instr_a)
            self.update_objs_poss(instr.instr_b)
        else:
            instr.update_objs_poss()

    def _gen_grid(self, width, height):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                super()._gen_grid(width, height)


                # Generate the mission
                self.gen_mission()

                # Validate the instructions
                self.validate_instrs(self.instrs)

            except RecursionError as error:
                print('Timeout during mission generation:', error)
                continue

            except RejectSampling as error:
                #print('Sampling rejected:', error)
                continue

            break

        # Generate the surface form for the instructions
        self.surface = self.instrs.surface(self)
        self.mission = self.surface

    def validate_instrs(self, instr):
        """
        Perform some validation on the generated instructions
        """
        # Gather the colors of locked doors
        if hasattr(self, 'unblocking') and self.unblocking:
            colors_of_locked_doors = []
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    room = self.get_room(i, j)
                    for door in room.doors:
                        if door and isinstance(door, LockedDoor):
                            colors_of_locked_doors.append(door.color)

        if isinstance(instr, PutNextInstr):
            # Resolve the objects referenced by the instruction
            instr.reset_verifier(self)

            # Check that the objects are not already next to each other
            if instr.objs_next():
                raise RejectSampling('objs already next to each other')

            # Check that we are not asking to move an object next to itself
            move = instr.desc_move
            fixed = instr.desc_fixed
            if len(move.obj_set) == 1 and len(fixed.obj_set) == 1:
                if move.obj_set[0] is fixed.obj_set[0]:
                    raise RejectSampling('cannot move an object next to itself')

        if isinstance(instr, ActionInstr):
            if not hasattr(self, 'unblocking') or not self.unblocking:
                return
            # TODO: either relax this a bit or make the bot handle this super corner-y scenarios
            # Check that the instruction doesn't involve a key that matches the color of a locked door
            potential_objects = ('desc', 'desc_move', 'desc_fixed')
            for attr in potential_objects:
                if hasattr(instr, attr):
                    obj = getattr(instr, attr)
                    if obj.type == 'key' and obj.color in colors_of_locked_doors:
                        raise RejectSampling('cannot do anything with/to a key that can be used to open a door')
            return

        if isinstance(instr, SeqInstr):
            self.validate_instrs(instr.instr_a)
            self.validate_instrs(instr.instr_b)
            return

        assert False, "unhandled instruction type"

    def gen_mission(self):
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError

    @property
    def level_name(self):
        return self.__class__.level_name

    @property
    def gym_id(self):
        return self.__class__.gym_id

    def num_navs_needed(self, instr):
        """
        Compute the maximum number of navigations needed to perform
        a simple or complex instruction
        """

        if isinstance(instr, PutNextInstr):
            return 2

        if isinstance(instr, ActionInstr):
            return 1

        if isinstance(instr, SeqInstr):
            na = self.num_navs_needed(instr.instr_a)
            nb = self.num_navs_needed(instr.instr_b)
            return na + nb

    def open_all_doors(self):
        """
        Open all the doors in the maze
        """

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        door.is_open = True

    def check_objs_reachable(self, raise_exc=True):
        """
        Check that all objects are reachable from the agent's starting
        position without requiring any other object to be moved
        (without unblocking)
        """

        # Reachable positions
        reachable = set()

        # Work list
        stack = [self.start_pos]

        while len(stack) > 0:
            i, j = stack.pop()

            if i < 0 or i >= self.grid.width or j < 0 or j >= self.grid.height:
                continue

            if (i, j) in reachable:
                continue

            # This position is reachable
            reachable.add((i, j))

            cell = self.grid.get(i, j)

            # If there is something other than a door in this cell, it
            # blocks reachability
            if cell and cell.type is not 'door':
                continue

            # Visit the horizontal and vertical neighbors
            stack.append((i+1, j))
            stack.append((i-1, j))
            stack.append((i, j+1))
            stack.append((i, j-1))

        # Check that all objects are reachable
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)

                if not cell or cell.type is 'wall':
                    continue

                if (i, j) not in reachable:
                    if not raise_exc:
                        return False
                    raise RejectSampling('unreachable object at ' + str((i, j)))

        # All objects reachable
        return True

class LevelGen(RoomGridLevel):
    """
    Level generator which attempts to produce every possible sentence in
    the baby language as an instruction.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        locked_room_prob=0.5,
        locations=True,
        unblocking=True,
        implicit_unlock=True,
        action_kinds=['goto', 'pickup', 'open', 'putnext'],
        instr_kinds=['action', 'and', 'seq'],
        seed=None,
        maxim_depth=None,
        config_file=None,
        instr_no_door=False
    ):

        self.num_dists = num_dists
        self.locked_room_prob = locked_room_prob
        self.locations = locations
        self.unblocking = unblocking
        self.implicit_unlock = implicit_unlock
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds

        self.locked_room = None
        ## Maxim depth is not the actual depth, meerly an upperbound on the depth
        self.maxim_depth = maxim_depth

        ## How to save the objects and distractors

        self.object_list = []
        self.distractor_list = None
        self.config_file = config_file
        self.instr_no_door = instr_no_door

        if self.instr_no_door:
            self.actions_kinds = [i for i in self.action_kinds if i != 'open']

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            seed=seed
        )


    def gen_mission(self):
        if self.config_file is not None:
            self.gen_mission_config(self.config_file)
        else:

            self.place_agent()

            if self._rand_float(0, 1) < self.locked_room_prob:
                self.add_locked_room()

            self.connect_all()
            self.empty_room_grid = deepcopy(self.room_grid)
            self.empty_grid = deepcopy(self.grid)

            self.distractor_list, self.objs_loc_list = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

            # If no unblocking required, make sure all objects are
            # reachable without unblocking
            if not self.unblocking:
                self.check_objs_reachable()

            # Generate random instructions
            self.instrs, self.instrs_tuple = self.rand_instr(
                action_kinds=self.action_kinds,
                instr_kinds=self.instr_kinds,
                maxim_depth=self.maxim_depth
            )


    def gen_mission_config(self, config_file):
        with open(config_file, 'rb') as f:
            task_config = pickle.load(f)

        if 'maxim_depth' in task_config:
            self.maxim_depth = task_config['maxim_depth']

        if 'room_grid' in task_config:
            self.grid = task_config['grid']
            self.room_grid = task_config['room_grid']
        self.empty_grid = deepcopy(self.grid)
        self.room_grid = deepcopy(self.room_grid)

        ## Initial point can be in the environemtn
        if 'start_room' in task_config:
            self.place_agent(task_config['start_room'][0], task_config['start_room'][1])
        else:
            self.place_agent()
        # self.place_agent()

        if 'objects' in task_config:
            self.distractor_list, self.objs_loc_list = self.add_distractors_config(task_config['objects'])
        else:
            self.distractor_list, self.objs_loc_list = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        self.connect_all()
        if not self.unblocking:
            self.check_objs_reachable()

        # Generate instructions
        self.instrs_tuple = task_config['instrs_tuple']
        self.maxim_depth = task_config['maxim_depth']
        self.instrs = self.spec_instr(self.instrs_tuple)

    def place_agent(self, i=None, j=None, rand_dir=True):
        """
        Place the agent in a room
        """

        if i == None:
            i = self._rand_int(0, self.num_cols)
        if j == None:
            j = self._rand_int(0, self.num_rows)

        room = self.room_grid[j][i]

        # Find a position that is not right in front of an object
        while True:
            MiniGridEnv.place_agent(self, room.top, room.size, rand_dir, max_tries=1000)
            pos = self.start_pos
            dir = DIR_TO_VEC[self.start_dir]
            front_pos = pos + dir
            front_cell = self.grid.get(*front_pos)
            if front_cell is None or front_cell.type is 'wall':
                break
        self.start_room = (i, j)
        return self.start_pos

    @staticmethod
    def find_object_instr(instruction, object_list):
        if instruction[0] == 'action':
            if instruction[1][0] == 'putnext':
                object_list.append(instruction[1][1][0])
                object_list.append(instruction[1][1][1])
            else:
                object_list.append(instruction[1][1])
        else:
            LevelGen.find_object_instr(instruction[1][0], object_list)
            LevelGen.find_object_instr(instruction[1][1], object_list)



    def save_mission(self, save_path,
                     fields=['num_dists', 'instr', 'maxim_depth']):
        results = dict()
        fields_list = ['locked_room_prob', 'locations', 'unblocking', 'implicit_unlock', 'action_kinds', 'instr_kinds', 'locked_room', 'room_size', 'num_rows', 'num_cols', 'actions', 'action_space', 'reward_range', 'grid_size', 'max_steps', 'see_through_walls', 'start_pos', 'agent_dir', 'start_pos']

        if 'maxim_depth' in fields:
            fields_list.append('maxim_depth')
        if 'start_room' in fields:
            fields_list.append('start_room')
        if 'num_dists' in fields:
            fields_list.append('num_dists')
        for field in fields_list:
            results[field] = self.__dict__[field]

        results['objects'] = []

        if 'instr' in fields:
            results['instrs_tuple'] = self.instrs_tuple
            object_list = []
            LevelGen.find_object_instr(self.instrs_tuple, object_list)
            for obj in object_list:
                results['objects'].append((obj.type, obj.color))

        ## TODO: recover the distractors

        if 'room_grid' in fields:
            results['room_grid'] = self.empty_room_grid
            results['grid'] = self.empty_grid

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)



    def add_distractors_config(self, objs):
        dists = []
        objs_loc = []
        for obj in objs:
            ## Obj should be a tuple (kind, color, i , j)
            if len(obj) == 2:
                i = self._rand_int(0, self.num_cols)
                j = self._rand_int(0, self.num_rows)
            elif len(obj) == 4:
                i = obj[2]
                j = obj[3]

            else:
                raise ValueError('lenth of object tuple must be 2, without location or 4, with location')

            objs_loc.append((objs[0], obj[1], i, j))
            # from pdb import set_trace as st
            # st()
            dist, _ = self.add_object(i, j, kind=obj[0], color=obj[1])
            dists.append(dist)
        dists_objs_loc = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        dists = dists + dists_objs_loc[0]
        objs_loc = objs_loc + dists_objs_loc[1]
        return dists, objs_loc

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True):
        """
        Add random objects that can potentially distract/confuse the agent.
        """

        # Collect a list of existing objects
        objs = []
        objs_loc = []
        for row_num, row in enumerate(self.room_grid):
            for room_num, room in enumerate(row):
                for obj in room.objs:
                    objs.append((obj.type, obj.color))
                    objs_loc.append((obj.type, obj.color, row_num, room_num))

        # List of distractors added
        dists = []


        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            type = self._rand_elem(['key', 'ball', 'box'])
            obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i == None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j == None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            objs_loc.append((type, color, room_i, room_j))
            dists.append(dist)

        return dists, objs_loc


    # def add_object(self, i, j, kind=None, color=None):
    #     """
    #     Add a new object to room (i, j)
    #     """
    #     if kind == None:
    #         kind = self._rand_elem(['key', 'ball', 'box'])
    #
    #     if color == None:
    #         color = self._rand_color()
    #
    #     assert kind in ['key', 'ball', 'box']
    #     if kind == 'key':
    #         obj = Key(color)
    #     elif kind == 'ball':
    #         obj = Ball(color)
    #     elif kind == 'box':
    #         obj = Box(color)
    #
    #     return self.place_in_room(i, j, obj)


    def add_locked_room(self):
        start_room = self.room_from_pos(*self.start_pos)

        # Until we've successfully added a locked room
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            door_idx = self._rand_int(0, 4)
            self.locked_room = self.get_room(i, j)

            # Don't lock the room the agent starts in
            if self.locked_room is start_room:
                continue

            # Don't add a locked door in an external wall
            if self.locked_room.neighbors[door_idx] is None:
                continue

            door, _ = self.add_door(
                i, j,
                door_idx,
                locked=True
            )

            # Done adding locked room
            break

        # Until we find a room to put the key
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            key_room = self.get_room(i, j)

            if key_room is self.locked_room:
                continue

            self.add_object(i, j, 'key', door.color)
            break

    def rand_obj(self, types=OBJ_TYPES, colors=COLOR_NAMES, max_tries=100, obj_cont=None):
        """
        Generate a random object descriptor
        """

        num_tries = 0

        # Keep trying until we find a matching object
        while True:
            if num_tries > max_tries:
                raise RecursionError('failed to find suitable object')
            num_tries += 1

            if obj_cont is None:
                color = self._rand_elem([None, *colors])
                type = self._rand_elem(types)
            else:
                color = obj_cont.color
                type = obj_cont.type

            loc = None
            if self.locations and self._rand_bool():
                loc = self._rand_elem(LOC_NAMES)
                # if obj_cont is not None and len(obj_cont) > 2:
                #     loc = obj_cont[3]

            desc = ObjDesc(type, color, loc)

            # Find all objects matching the descriptor
            objs, poss = desc.find_matching_objs(self)

            # The description must match at least one object
            if len(objs) == 0:
                continue

            # If no implicit unlocking is required
            if not self.implicit_unlock and self.locked_room:
                # Check that at least one object is not in the locked room
                pos_not_locked = list(filter(
                    lambda p: not self.locked_room.pos_inside(*p),
                    poss
                ))

                if len(pos_not_locked) == 0:
                    continue

            # Found a valid object description
            return desc

    def spec_instr(self, instr, depth=0):
        """
        Generate instructions as specifed
        :param action_kind:
        :param instr_kind:
        :param depth:
        :param obj:
        :return:
        """
        assert len(instr) == 2
        kind = instr[0]
        content = instr[1]
        if kind == 'action':
            action = content[0]
            obj_cont = content[1]

            if action == 'goto':
                return GoToInstr(self.rand_obj(obj_cont=obj_cont))
            elif action == 'pickup':
                return PickupInstr(self.rand_obj(obj_cont=obj_cont))
            elif action == 'open':
                assert obj_cont[0] == 'door'
                return OpenInstr(self.rand_obj(obj_cont=obj_cont))
            elif action == 'putnext':
                return PutNextInstr(
                    self.rand_obj(obj_cont=obj_cont[0]),
                    self.rand_obj(obj_cont=obj_cont[1])
                )

            assert False

        elif kind == 'and':
            instr_a = self.spec_instr(
                content[0],
                depth=depth + 1
            )
            instr_b = self.spec_instr(
                content[1],
                depth=depth + 1
            )
            return AndInstr(instr_a, instr_b)

        elif kind == 'before' or kind == 'after':
            instr_a = self.spec_instr(
                content[0],
                depth=depth + 1
            )
            instr_b = self.spec_instr(
                content[1],
                depth=depth + 1
            )
            if kind == 'before':
                return BeforeInstr(instr_a, instr_b)
            elif kind == 'after':
                return AfterInstr(instr_a, instr_b)

            assert False

        assert False

    def rand_instr(
        self,
        action_kinds,
        instr_kinds,
        depth=0,
        obj=None,
        maxim_depth=None
    ):
        """
        Generate random instructions
        """

        kind = self._rand_elem(instr_kinds)
        if maxim_depth is not None and depth >= maxim_depth:
            kind = 'action'
        if self.maxim_depth is None:
            self.maxim_depth = depth
        else:
            self.maxim_depth = max(self.maxim_depth, depth)


        if kind is 'action':
            action = self._rand_elem(action_kinds)
            if action is 'goto':
                if self.instr_no_door:
                    obj = self.rand_obj(types=OBJ_TYPES_NOT_DOOR)
                else:
                    obj = self.rand_obj()
                return GoToInstr(obj), ('action', ('goto', deepcopy(obj)))
            elif action is 'pickup':
                obj = self.rand_obj(types=OBJ_TYPES_NOT_DOOR)
                return PickupInstr(obj), ('action', ('goto', deepcopy(obj)))
            elif action is 'open':
                obj = self.rand_obj(types=['door'])
                return OpenInstr(obj), ('action', ('open', deepcopy(obj)))
            elif action is 'putnext':
                obj1 = self.rand_obj(types=OBJ_TYPES_NOT_DOOR)
                if self.instr_no_door:
                    obj2 = self.rand_obj(types=OBJ_TYPES_NOT_DOOR)
                else:
                    obj2 = self.rand_obj()
                return PutNextInstr(
                    obj1,
                    obj2,
                ), ('action', ('putnext', (deepcopy(obj1), deepcopy(obj2))))

            assert False

        elif kind is 'and':
            instr_a, instr_a_tuple = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth+1,
                maxim_depth=maxim_depth
            )
            instr_b, instr_b_tuple = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth+1,
                maxim_depth=maxim_depth
            )
            return AndInstr(instr_a, instr_b), ('and', (instr_a_tuple, instr_b_tuple))

        elif kind is 'seq':
            instr_a, instr_a_tuple = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth+1,
                maxim_depth=maxim_depth
            )
            instr_b, instr_b_tuple = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth+1,
                maxim_depth=maxim_depth
            )

            kind = self._rand_elem(['before', 'after'])

            if kind is 'before':
                return BeforeInstr(instr_a, instr_b), ('before', (instr_a_tuple, instr_b_tuple))
            elif kind is 'after':
                return AfterInstr(instr_a, instr_b), ('after', (instr_a_tuple, instr_b_tuple))

            assert False

        assert False


# Dictionary of levels, indexed by name, lexically sorted
level_dict = OrderedDict()


def register_levels(module_name, globals):
    """
    Register OpenAI gym environments for all levels in a file
    """
    # Iterate through global names
    for global_name in sorted(list(globals.keys())):
        if not global_name.startswith('Level_'):
            continue

        level_name = global_name.split('Level_')[-1]
        level_class = globals[global_name]

        # Register the levels with OpenAI Gym
        gym_id = 'BabyAI-%s-v0' % (level_name)
        entry_point = '%s:%s' % (module_name, global_name)
        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        # Add the level to the dictionary
        level_dict[level_name] = level_class

        # Store the name and gym id on the level class
        level_class.level_name = level_name
        level_class.gym_id = gym_id


def test():
    for idx, level_name in enumerate(level_dict.keys()):
        print('Level %s (%d/%d)' % (level_name, idx+1, len(level_dict)))

        level = level_dict[level_name]

        # Run the mission for a few episodes
        rng = random.Random(0)
        num_episodes = 0
        for i in range(0, 15):
            mission = level(seed=i)

            # Check that the surface form was generated
            assert isinstance(mission.surface, str)
            assert len(mission.surface) > 0
            obs = mission.reset()
            assert obs['mission'] == mission.surface

            # Reduce max_steps because otherwise tests take too long
            mission.max_steps = min(mission.max_steps, 200)

            # Check for some known invalid patterns in the surface form
            import re
            surface = mission.surface
            assert not re.match(r".*pick up the [^ ]*door.*", surface), surface

            while True:
                action = rng.randint(0, mission.action_space.n - 1)
                obs, reward, done, info = mission.step(action)
                if done:
                    obs = mission.reset()
                    break

            num_episodes += 1

        # The same seed should always yield the same mission
        m0 = level(seed=0)
        m1 = level(seed=0)
        grid1 = m0.unwrapped.grid
        grid2 = m1.unwrapped.grid
        assert grid1 == grid2
        assert m0.surface == m1.surface

    # Check that gym environment names were registered correctly
    gym.make('BabyAI-1RoomS8-v0')
    gym.make('BabyAI-BossLevel-v0')
