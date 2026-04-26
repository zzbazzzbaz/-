#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Standard PPO algorithm for Robot Vacuum.
清扫大作战 PPO 算法。

Loss composition / 损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss
"""

import os
import time

import torch
import torch.nn.functional as F

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in optimizer.param_groups for p in pg["params"]]
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.label_size = Config.ACTION_NUM

        self.train_step = 0
        self.last_report_time = 0

    def learn(self, list_sample_data):
        """Training entry: perform one PPO gradient step on a batch of SampleData.

        训练入口：接收一批 SampleData，执行一步梯度更新。
        """
        obs = torch.stack([s.obs for s in list_sample_data]).to(self.device)
        legal_action = torch.stack([s.legal_action for s in list_sample_data]).to(self.device)
        act = torch.stack([s.act for s in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([s.prob for s in list_sample_data]).to(self.device)
        old_value = torch.stack([s.value for s in list_sample_data]).to(self.device)
        reward_sum = torch.stack([s.reward_sum for s in list_sample_data]).to(self.device)
        advantage = torch.stack([s.advantage for s in list_sample_data]).to(self.device)
        reward = torch.stack([s.reward for s in list_sample_data]).to(self.device)

        if Config.NORMALIZE_ADVANTAGE and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        self.model.set_train_mode()
        batch_size = obs.shape[0]
        mini_batch_size = min(Config.MINI_BATCH_SIZE, batch_size)
        stat_sum = {
            "total_loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        stat_count = 0

        for _ in range(Config.PPO_EPOCHS):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mini_batch_size):
                mb_idx = indices[start : start + mini_batch_size]

                rst_list = self.model(obs[mb_idx])
                logits, value_pred = rst_list[0], rst_list[1]

                total_loss, info = self._compute_loss(
                    logits=logits,
                    value_pred=value_pred,
                    legal_action=legal_action[mb_idx],
                    old_action=act[mb_idx],
                    old_prob=old_prob[mb_idx],
                    old_value=old_value[mb_idx],
                    reward_sum=reward_sum[mb_idx],
                    advantage=advantage[mb_idx],
                )

                self.optimizer.zero_grad()
                total_loss.backward()

                if Config.USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

                self.optimizer.step()
                self.train_step += 1

                for key in stat_sum:
                    stat_sum[key] += info[key]
                stat_count += 1

            if stat_count > 0 and stat_sum["approx_kl"] / stat_count > Config.TARGET_KL:
                break

        info = {key: value / max(stat_count, 1) for key, value in stat_sum.items()}
        results = {"total_loss": info["total_loss"]}

        # Periodic monitoring report
        # 定期上报监控
        now = time.time()
        if now - self.last_report_time >= 60:
            results["value_loss"] = round(info["value_loss"], 4)
            results["policy_loss"] = round(info["policy_loss"], 4)
            results["entropy_loss"] = round(info["entropy_loss"], 4)
            results["reward"] = round(reward.mean().item(), 4)
            results["approx_kl"] = round(info["approx_kl"], 4)
            results["clip_fraction"] = round(info["clip_fraction"], 4)

            self.logger.info(
                f"policy_loss: {results['policy_loss']}, "
                f"value_loss: {results['value_loss']}, "
                f"entropy_loss: {results['entropy_loss']}, "
                f"approx_kl: {results['approx_kl']}, "
                f"clip_fraction: {results['clip_fraction']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_time = now

        return results

    def _compute_loss(self, logits, value_pred, legal_action, old_action, old_prob, old_value, reward_sum, advantage):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 三项损失。
        """
        # Value loss (clipped)
        # 价值损失（裁剪）
        tdret = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        vp = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred
        ov = old_value.squeeze(-1) if old_value.dim() > 1 else old_value

        vp_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                F.smooth_l1_loss(vp, tdret, reduction="none"),
                F.smooth_l1_loss(vp_clip, tdret, reduction="none"),
            ).mean()
        )

        # Policy loss (PPO clip)
        # 策略损失（PPO clip）
        prob_dist = self._masked_softmax(logits, legal_action)
        entropy_loss = (-(prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1)).mean()

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True)

        ratio = new_prob / old_action_prob.clamp(1e-9)
        log_ratio = torch.log(new_prob.clamp_min(1e-9)) - torch.log(old_action_prob.clamp_min(1e-9))
        approx_kl = (-log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()

        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = adv.unsqueeze(-1)

        policy_loss = torch.maximum(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        # Total loss
        # 总损失
        entropy_beta = self._entropy_beta()
        total_loss = self.vf_coef * value_loss + policy_loss - entropy_beta * entropy_loss

        return total_loss, {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }

    def _masked_softmax(self, logits, legal_action):
        """Apply legal action mask to logits before computing softmax.

        对 logits 应用合法动作掩码后计算 softmax。
        """
        legal_mask = legal_action > 0.5
        safe_logits = logits.masked_fill(~legal_mask, -1e9)
        return F.softmax(safe_logits, dim=1)

    def _entropy_beta(self):
        """Linearly decay entropy regularization for fast early exploration."""
        progress = min(float(self.train_step) / max(Config.BETA_DECAY_STEPS, 1), 1.0)
        return Config.BETA_START + progress * (Config.BETA_END - Config.BETA_START)
