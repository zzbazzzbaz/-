#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _signed_norm(v, v_max):
    """Normalize signed value to [-1, 1]."""
    if v_max <= 0:
        return 0.0
    return float(np.clip(float(v) / float(v_max), -1.0, 1.0))


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 5  # Cropped view radius (11×11) / 裁剪后的视野半径
    ACTION_DIRS = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)
        self.prev_pos = None
        self.has_position_history = False
        self.current_visit_count = 0
        self.is_new_cell = False
        self.last_action = -1

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1
        self.total_score = 0
        self.clean_score = 0
        self.step_cleaned_count = 0
        self.max_step = 1000

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0
        self.visit_count_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint16)

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8
        self.terminated = False
        self.truncated = False

        self.remaining_charge = 0
        self.prev_battery = 600
        self.prev_battery_max = 600
        self.prev_on_charger = False
        self.prev_low_battery = False
        self.was_recharge_mode = False
        self.charge_count = 0
        self.last_charge_count = 0
        self.charge_delta = 0
        self.nearest_charger_dx = 0.0
        self.nearest_charger_dz = 0.0
        self.nearest_charger_center_dx = 0.0
        self.nearest_charger_center_dz = 0.0
        self.nearest_charger_dist = float(self.GRID_SIZE)
        self.nearest_charger_range_dist = float(self.GRID_SIZE)
        self.last_nearest_charger_range_dist = float(self.GRID_SIZE)
        self.battery_margin = 0.0
        self.has_charger = False
        self.low_battery = False
        self.on_charger = False
        self.charger_rects = []
        self.recharge_mode = False
        self.recharge_steps = 0

        self.nearest_npc_dx = 0.0
        self.nearest_npc_dz = 0.0
        self.nearest_npc_dist = float(self.GRID_SIZE)
        self.npc_danger = False
        self.npcs = []

        self.local_dirt_ratio = 0.0
        self.local_obstacle_ratio = 0.0

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation.get("frame_state", {})
        extra_frame_state = env_obs.get("extra_info", {}).get("frame_state", {})
        env_info = observation.get("env_info", {})
        hero = frame_state.get("heroes", {})
        if isinstance(hero, list):
            hero = hero[0] if hero else {}

        self.last_action = int(last_action)
        self.step_no = int(observation.get("step_no", env_info.get("step_no", self.step_no)))
        self.terminated = bool(env_obs.get("terminated", False))
        self.truncated = bool(env_obs.get("truncated", False))
        self.prev_battery = self.battery
        self.prev_battery_max = self.battery_max
        self.prev_on_charger = self.on_charger
        self.prev_low_battery = self.low_battery
        self.was_recharge_mode = self.recharge_mode
        self.prev_pos = self.cur_pos if self.has_position_history else None
        hero_pos = hero.get("pos") or env_info.get("pos") or {"x": self.cur_pos[0], "z": self.cur_pos[1]}
        self.cur_pos = (int(hero_pos.get("x", self.cur_pos[0])), int(hero_pos.get("z", self.cur_pos[1])))
        self.has_position_history = True

        hx, hz = self.cur_pos
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            self.current_visit_count = int(self.visit_count_map[hx, hz])
            self.is_new_cell = self.current_visit_count == 0
            self.visit_count_map[hx, hz] = min(self.current_visit_count + 1, np.iinfo(np.uint16).max)
        else:
            self.current_visit_count = 0
            self.is_new_cell = False

        # Battery / 电量
        self.battery = int(hero.get("battery", env_info.get("remaining_charge", self.battery)))
        self.battery_max = max(int(hero.get("battery_max", env_info.get("battery_max", self.battery_max))), 1)
        self.remaining_charge = int(env_info.get("remaining_charge", self.battery))
        self.max_step = max(int(env_info.get("max_step", self.max_step)), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero.get("dirt_cleaned", env_info.get("clean_score", self.dirt_cleaned)))
        self.total_dirt = max(int(env_info.get("total_dirt", self.total_dirt)), 1)
        self.total_score = int(env_info.get("total_score", self.total_score))
        self.clean_score = int(env_info.get("clean_score", self.dirt_cleaned))
        step_cleaned_cells = env_info.get("step_cleaned_cells") or []
        self.step_cleaned_count = len(step_cleaned_cells)

        # Charge progress / 充电进度
        self.last_charge_count = self.charge_count
        self.charge_count = int(env_info.get("charge_count", self.charge_count))
        self.charge_delta = max(0, self.charge_count - self.last_charge_count)

        # Legal actions / 合法动作
        raw_legal_act = observation.get("legal_action") or observation.get("legal_act") or [1] * 8
        self._legal_act = [int(x) for x in raw_legal_act[:8]]
        if len(self._legal_act) < 8:
            self._legal_act.extend([1] * (8 - len(self._legal_act)))

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)
            self._update_local_map_stats()

        organs = frame_state.get("organs") or extra_frame_state.get("organs") or []
        npcs = frame_state.get("npcs") or extra_frame_state.get("npcs") or []
        self.npcs = list(npcs)
        self._update_charger_state(hx, hz, organs)
        self._update_npc_state(hx, hz, self.npcs)
        self._update_recharge_mode()

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _update_local_map_stats(self):
        """Cache coarse 21x21 map statistics."""
        view = self._view_map
        if view is None or view.size == 0:
            self.local_dirt_ratio = 0.0
            self.local_obstacle_ratio = 0.0
            return
        total = float(view.size)
        self.local_dirt_ratio = float(np.sum(view == 2) / total)
        self.local_obstacle_ratio = float(np.sum(view == 0) / total)

    def _update_charger_state(self, hx, hz, organs):
        """Find nearest charger and cache distance/direction features."""
        self.last_nearest_charger_range_dist = self.nearest_charger_range_dist
        self.has_charger = False
        self.on_charger = False
        self.nearest_charger_dx = 0.0
        self.nearest_charger_dz = 0.0
        self.nearest_charger_center_dx = 0.0
        self.nearest_charger_center_dz = 0.0
        self.nearest_charger_dist = float(self.GRID_SIZE)
        self.nearest_charger_range_dist = float(self.GRID_SIZE)
        self.charger_rects = []

        best = None
        for organ in organs:
            if not isinstance(organ, dict):
                continue
            if int(organ.get("sub_type", 1)) != 1:
                continue
            pos = organ.get("pos") or {}
            ox = int(pos.get("x", 0))
            oz = int(pos.get("z", 0))
            w = max(int(organ.get("w", 3)), 1)
            h = max(int(organ.get("h", 3)), 1)

            for rx, rz in ((ox, oz), (ox - w // 2, oz - h // 2)):
                self.charger_rects.append((rx, rz, w, h))
                dx, dz = self._relative_vector_to_rect(hx, hz, rx, rz, w, h)
                dist = float(np.sqrt(dx * dx + dz * dz))
                range_dist = float(max(abs(dx), abs(dz)))
                center_dx = (rx + (w - 1) * 0.5) - hx
                center_dz = (rz + (h - 1) * 0.5) - hz
                if best is None or range_dist < best[0] or (range_dist == best[0] and dist < best[1]):
                    best = (range_dist, dist, dx, dz, center_dx, center_dz)

        if best is None:
            self.battery_margin = float(self.battery)
            return

        range_dist, dist, dx, dz, center_dx, center_dz = best
        self.has_charger = True
        self.nearest_charger_dx = float(dx)
        self.nearest_charger_dz = float(dz)
        self.nearest_charger_center_dx = float(center_dx)
        self.nearest_charger_center_dz = float(center_dz)
        self.nearest_charger_dist = float(dist)
        self.nearest_charger_range_dist = float(range_dist)
        self.on_charger = range_dist <= 0.0
        self.battery_margin = float(self.battery) - self.nearest_charger_range_dist

    def _relative_vector_to_rect(self, x, z, rx, rz, w, h):
        """Relative vector from point to the nearest cell in a rectangle."""
        if x < rx:
            dx = rx - x
        elif x > rx + w - 1:
            dx = x - (rx + w - 1)
        else:
            dx = 0

        if z < rz:
            dz = rz - z
        elif z > rz + h - 1:
            dz = z - (rz + h - 1)
        else:
            dz = 0
        return float(dx), float(dz)

    def _update_npc_state(self, hx, hz, npcs):
        """Find nearest NPC and cache safety features."""
        self.nearest_npc_dx = 0.0
        self.nearest_npc_dz = 0.0
        self.nearest_npc_dist = float(self.GRID_SIZE)
        self.npc_danger = False

        best = None
        for npc in npcs:
            if not isinstance(npc, dict):
                continue
            pos = npc.get("pos") or {}
            nx = int(pos.get("x", 0))
            nz = int(pos.get("z", 0))
            dx = nx - hx
            dz = nz - hz
            cheb = float(max(abs(dx), abs(dz)))
            if best is None or cheb < best[0]:
                best = (cheb, dx, dz)

        if best is None:
            return

        cheb, dx, dz = best
        self.nearest_npc_dx = float(dx)
        self.nearest_npc_dz = float(dz)
        self.nearest_npc_dist = float(cheb)
        self.npc_danger = abs(dx) <= 1 and abs(dz) <= 1

    def _update_recharge_mode(self):
        """Enter/exit low-battery recharge mode."""
        battery_ratio = self.battery / max(self.battery_max, 1)
        self.low_battery = battery_ratio < 0.35

        if not self.has_charger:
            self.recharge_mode = False
            return

        if self.charge_delta > 0 or (self.on_charger and battery_ratio > 0.85):
            self.recharge_mode = False
        elif self.battery <= self.nearest_charger_range_dist + 18 or battery_ratio < 0.22:
            self.recharge_mode = True
        elif self.recharge_mode and battery_ratio < 0.85:
            self.recharge_mode = True
        else:
            self.recharge_mode = False

        if self.recharge_mode:
            self.recharge_steps += 1

    def _min_charger_range_dist(self, x, z):
        if not self.charger_rects:
            return float(self.GRID_SIZE)
        dists = []
        for rx, rz, w, h in self.charger_rects:
            dx, dz = self._relative_vector_to_rect(x, z, rx, rz, w, h)
            dists.append(max(abs(dx), abs(dz)))
        return float(min(dists))

    def _get_local_view_feature(self):
        """Local view feature (121D): crop center 11×11 from 21×21.

        局部视野特征（121D）：从 21×21 视野中心裁剪 11×11。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature (28D).

        全局状态特征（28D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
          [12] charger_dx        nearest charger x direction / 最近充电桩 x 相对方向
          [13] charger_dz        nearest charger z direction / 最近充电桩 z 相对方向
          [14] charger_dist      nearest charger distance / 最近充电桩距离
          [15] battery_margin    battery minus charger distance / 电量安全余量
          [16] low_battery       low-battery flag / 低电量标记
          [17] recharge_mode     recharge-mode flag / 回充模式标记
          [18] on_charger        on charger flag / 是否在充电桩范围
          [19] charge_delta      charge count increased / 本步是否成功充电
          [20] npc_dx            nearest NPC x direction / 最近 NPC x 相对方向
          [21] npc_dz            nearest NPC z direction / 最近 NPC z 相对方向
          [22] npc_dist          nearest NPC Chebyshev distance / 最近 NPC 切比雪夫距离
          [23] npc_danger        in NPC 3x3 danger zone / 是否处于 NPC 3x3 危险区
          [24] local_dirt_ratio  dirt ratio in 21x21 view / 21x21 视野污渍比例
          [25] obstacle_ratio    obstacle ratio in 21x21 view / 21x21 视野障碍比例
          [26] visit_count       current cell visit count / 当前格访问次数
          [27] step_cleaned      cells cleaned this step / 本步清扫格子数
        """
        step_norm = _norm(self.step_no, self.max_step)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        charge_delta = 1.0 if self.charge_delta > 0 else 0.0
        battery_margin_norm = _signed_norm(self.battery_margin, self.battery_max)
        visit_count_norm = _norm(min(self.current_visit_count, 10), 10)
        step_cleaned_norm = _norm(self.step_cleaned_count, 9)

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
                _signed_norm(self.nearest_charger_dx, self.GRID_SIZE),
                _signed_norm(self.nearest_charger_dz, self.GRID_SIZE),
                _norm(self.nearest_charger_range_dist, self.GRID_SIZE),
                battery_margin_norm,
                1.0 if self.low_battery else 0.0,
                1.0 if self.recharge_mode else 0.0,
                1.0 if self.on_charger else 0.0,
                charge_delta,
                _signed_norm(self.nearest_npc_dx, 20),
                _signed_norm(self.nearest_npc_dz, 20),
                _norm(self.nearest_npc_dist, 20),
                1.0 if self.npc_danger else 0.0,
                self.local_dirt_ratio,
                self.local_obstacle_ratio,
                visit_count_norm,
                step_cleaned_norm,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        legal = self._filter_blocked_actions(self._legal_act)
        legal = self._filter_npc_danger_actions(legal)
        if self.recharge_mode:
            legal = self._filter_recharge_actions(legal)
        elif self.on_charger and self.battery / max(self.battery_max, 1) > 0.65:
            legal = self._filter_leave_charger_actions(legal)
        return list(legal)

    def _filter_blocked_actions(self, legal_action):
        """Filter actions that are visibly blocked in the 21x21 view."""
        legal = [int(x) for x in legal_action]
        hx, hz = self.cur_pos
        for action, (dx, dz) in enumerate(self.ACTION_DIRS):
            if legal[action] <= 0:
                continue
            if not self._is_visible_cell_passable(dx, dz):
                legal[action] = 0
                continue
            if dx != 0 and dz != 0:
                side_a = self._is_visible_cell_passable(dx, 0)
                side_b = self._is_visible_cell_passable(0, dz)
                if not (side_a or side_b):
                    legal[action] = 0
            nx, nz = hx + dx, hz + dz
            if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                legal[action] = 0

        return legal if any(legal) else [int(x) for x in legal_action]

    def _is_visible_cell_passable(self, dx, dz):
        """Whether a relative 21x21-view cell is passable."""
        ri = self.VIEW_HALF + dx
        ci = self.VIEW_HALF + dz
        if not (0 <= ri < self._view_map.shape[0] and 0 <= ci < self._view_map.shape[1]):
            return True
        return int(self._view_map[ri, ci]) != 0

    def _filter_npc_danger_actions(self, legal_action):
        """Avoid actions that would enter any NPC 3x3 danger zone."""
        if not self.npcs:
            return list(legal_action)

        hx, hz = self.cur_pos
        safe = [int(x) for x in legal_action]
        for action, (dx, dz) in enumerate(self.ACTION_DIRS):
            if safe[action] <= 0:
                continue
            nx, nz = hx + dx, hz + dz
            if self._is_npc_danger_cell(nx, nz):
                safe[action] = 0

        return safe if any(safe) else list(legal_action)

    def _is_npc_danger_cell(self, x, z):
        for npc in self.npcs:
            if not isinstance(npc, dict):
                continue
            pos = npc.get("pos") or {}
            nx = int(pos.get("x", -999))
            nz = int(pos.get("z", -999))
            if abs(x - nx) <= 1 and abs(z - nz) <= 1:
                return True
        return False

    def _filter_recharge_actions(self, legal_action):
        """Restrict low-battery actions to moves that approach the charger."""
        if not self.has_charger:
            return list(legal_action)

        hx, hz = self.cur_pos
        current_dist = max(abs(self.nearest_charger_dx), abs(self.nearest_charger_dz))
        scored = []
        for action, (dx, dz) in enumerate(self.ACTION_DIRS):
            if legal_action[action] <= 0:
                continue
            next_dx = self.nearest_charger_dx - dx
            next_dz = self.nearest_charger_dz - dz
            next_dist = max(abs(next_dx), abs(next_dz))
            improvement = current_dist - next_dist
            alignment = dx * self.nearest_charger_dx + dz * self.nearest_charger_dz
            scored.append((improvement, alignment, action))

        if not scored:
            return list(legal_action)

        best_improvement = max(item[0] for item in scored)
        recharge = [0] * 8
        if best_improvement > 0:
            for improvement, _, action in scored:
                if improvement >= best_improvement - 0.1:
                    recharge[action] = 1
        else:
            best_alignment = max(item[1] for item in scored)
            for _, alignment, action in scored:
                if alignment >= best_alignment:
                    recharge[action] = 1

        return recharge if any(recharge) else list(legal_action)

    def _filter_leave_charger_actions(self, legal_action):
        """Prefer moves that leave charger range when battery is healthy."""
        if not self.has_charger:
            return list(legal_action)

        hx, hz = self.cur_pos
        current_dist = self._min_charger_range_dist(hx, hz)
        scored = []
        for action, (dx, dz) in enumerate(self.ACTION_DIRS):
            if legal_action[action] <= 0:
                continue
            nx, nz = hx + dx, hz + dz
            next_dist = self._min_charger_range_dist(nx, nz)
            away_score = -(dx * self.nearest_charger_center_dx + dz * self.nearest_charger_center_dz)
            scored.append((next_dist - current_dist, away_score, action))

        if not scored:
            return list(legal_action)

        best_escape = max(item[0] for item in scored)
        leave = [0] * 8
        if best_escape > 0:
            for escape, _, action in scored:
                if escape >= best_escape - 0.1:
                    leave[action] = 1
        else:
            best_away = max(item[1] for item in scored)
            for _, away_score, action in scored:
                if away_score >= best_away:
                    leave[action] = 1

        return leave if any(leave) else list(legal_action)

    def feature_process(self, env_obs, last_action):
        """Generate feature vector, legal action mask, and scalar reward.

        生成特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 121D
        global_state = self._get_global_state_feature()  # 28D
        legal_action = self.get_legal_action()  # 8D

        last_action_feature = np.zeros(8, dtype=np.float32)
        if 0 <= last_action < 8:
            last_action_feature[last_action] = 1.0

        feature = np.concatenate([local_view, global_state, last_action_feature])

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        # Cleaning reward / 清扫奖励
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaned_cells = self.step_cleaned_count if self.step_cleaned_count > 0 else cleaned_this_step
        cleaning_reward = 0.7 * cleaned_cells

        # Step penalty / 时间惩罚
        step_penalty = -0.002

        # Dense guidance: prefer moving toward visible dirt.
        # 稠密引导：鼓励向视野内污渍靠近。
        approach_reward = 0.0
        if not self.recharge_mode and (self.last_nearest_dirt_dist < 200.0 or self.nearest_dirt_dist < 200.0):
            dist_delta = float(np.clip(self.last_nearest_dirt_dist - self.nearest_dirt_dist, -5.0, 5.0))
            approach_reward = 0.01 * dist_delta if dist_delta > 0 else 0.006 * dist_delta

        # Recharge guidance only activates when battery safety is the bottleneck.
        # 仅在低电量/回充模式下引导靠近充电桩，避免高电量蹲充电桩。
        charge_reward = 0.0
        battery_ratio = self.battery / max(self.battery_max, 1)
        prev_battery_ratio = self.prev_battery / max(self.prev_battery_max, 1)
        useful_charge = self.charge_delta > 0 and (
            self.prev_low_battery or self.was_recharge_mode or prev_battery_ratio < 0.45
        )
        if useful_charge:
            charge_reward += 1.0
        elif self.charge_delta > 0 and battery_ratio > 0.65:
            charge_reward -= 0.25 * min(self.charge_delta, 3)

        if self.has_charger and (self.recharge_mode or self.low_battery):
            dist_delta = float(
                np.clip(self.last_nearest_charger_range_dist - self.nearest_charger_range_dist, -4.0, 4.0)
            )
            charge_reward += 0.04 * dist_delta if dist_delta > 0 else 0.02 * dist_delta
            if self.battery_margin < 0:
                charge_reward -= min(0.25, abs(self.battery_margin) / max(self.battery_max, 1))
        elif self.on_charger and battery_ratio > 0.65:
            charge_reward -= 0.08

        # Encourage covering new passable cells and mildly discourage loops.
        # 鼓励探索新格子，轻微惩罚反复绕圈。
        exploration_reward = 0.004 if self.is_new_cell else -0.0015 * min(self.current_visit_count, 6)

        # Collision/stuck signal: invalid moves waste both step and battery.
        # 撞墙/原地不动会浪费步数和电量。
        stuck_penalty = 0.0
        if self.prev_pos is not None and self.cur_pos == self.prev_pos and 0 <= self.last_action < 8:
            stuck_penalty = -0.03

        npc_penalty = 0.0
        if self.npc_danger:
            npc_penalty -= 4.0
        elif self.nearest_npc_dist <= 3:
            npc_penalty -= 0.05 * (4 - self.nearest_npc_dist)

        terminal_penalty = 0.0
        if self.terminated and not self.truncated:
            if self.battery <= 0 or self.remaining_charge <= 0:
                terminal_penalty -= 4.0
            elif self.npc_danger or self.nearest_npc_dist <= 1:
                terminal_penalty -= 3.0

        return (
            cleaning_reward
            + approach_reward
            + charge_reward
            + exploration_reward
            + stuck_penalty
            + npc_penalty
            + terminal_penalty
            + step_penalty
        )
