import copy
import torch

import os
from actor import Actor
from critic import Critic, ValueCritic


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        expectile,
        discount,
        tau,
        temperature,
    ):

        self.actor = Actor(state_dim, action_dim, 256, 3).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.value = ValueCritic(state_dim, 256, 3).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.temperature = temperature

        self.total_it = 0
        self.expectile = expectile

    def update_v(self, states, actions, logger=None):
        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2).detach()

        v = self.value(states)
        value_loss = loss(q - v, self.expectile).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        logger.log('train/value_loss', value_loss, self.total_it)
        logger.log('train/v', v.mean(), self.total_it)

    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():
            next_v = self.value(next_states)
            target_q = (rewards + self.discount * not_dones * next_v).detach()

        q1, q2 = self.critic(states, actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logger.log('train/critic_loss', critic_loss, self.total_it)
        logger.log('train/q1', q1.mean(), self.total_it)
        logger.log('train/q2', q2.mean(), self.total_it)

    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor(self, states, actions, logger=None):
        with torch.no_grad():
            v = self.value(states)
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2)
            exp_a = torch.exp((q - v) * self.temperature)
            exp_a = torch.clamp(exp_a, max=100.0).squeeze(-1).detach()

        mu = self.actor(states)
        actor_loss = (exp_a.unsqueeze(-1) * ((mu - actions)**2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        logger.log('train/actor_loss', actor_loss, self.total_it)
        logger.log('train/adv', (q - v).mean(), self.total_it)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor.get_action(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Update
        self.update_v(state, action, logger)
        self.update_actor(state, action, logger)
        self.update_q(state, action, reward, next_state, not_done, logger)
        self.update_target()

    def save(self, model_dir):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.total_it)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.total_it)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(self.total_it)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.total_it)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(self.total_it)}.pth"))
        torch.save(self.actor_scheduler.state_dict(), os.path.join(
            model_dir, f"actor_scheduler_s{str(self.total_it)}.pth"))

        torch.save(self.value.state_dict(), os.path.join(model_dir, f"value_s{str(self.total_it)}.pth"))
        torch.save(self.value_optimizer.state_dict(), os.path.join(
            model_dir, f"value_optimizer_s{str(self.total_it)}.pth"))
