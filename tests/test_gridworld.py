import numpy as np
import torch

from macarico import Actor
from macarico.actors import BOWActor
from macarico.annealing import EWMA
from macarico.features.sequence import AttendAt
from macarico.lts.reinforce import Reinforce
from macarico.policies.linear import SoftmaxPolicy
import macarico.tasks.gridworld as gw
import macarico.util


macarico.util.reseed()


def run_gridworld(env: gw.GridWorld, actor: Actor) -> None:
    policy = SoftmaxPolicy(actor, env.n_actions)
    baseline = EWMA(0.8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    loss_fn = gw.GridLoss()
    learner = Reinforce(policy, baseline)
    n_epochs = 5000
    losses, objs = [], []
    best_loss = None
    for epoch in range(1, 1+n_epochs):
        optimizer.zero_grad()
        env.run_episode(learner)
        loss_val = loss_fn(env.example)
        obj = learner.get_objective(loss_val)
        if not isinstance(obj, float):
            obj.backward()
            optimizer.step()
            obj = obj.item()

        losses.append(loss_val)
        objs.append(obj)
        if epoch % 1000 == 0 or epoch == n_epochs:
            loss = np.mean(losses[-500:])
            if best_loss is None or loss < best_loss[0]:
                best_loss = (loss, epoch)
            print(epoch, 'losses', loss, 'objective', np.mean(objs[-500:]),
                  'best_loss', best_loss, 'init_losses', np.mean(losses[:1000]))


def test0():
    print('\n===\n=== test0: p_step_success=1.0\n===')
    env = gw.make_default_gridworld(p_step_success=1.0)
    features = gw.GlobalGridFeatures(env.settings.width, env.settings.height)
    attention = AttendAt(features, position=lambda _: 0)
    actor = BOWActor([attention], env.n_actions)
    run_gridworld(env, actor)


def test1():
    print('\n===\n=== test1: p_step_success=0.8\n===')
    env = gw.make_default_gridworld(p_step_success=0.8)
    features = gw.GlobalGridFeatures(env.settings.width, env.settings.height)
    attention = AttendAt(features, position=lambda _: 0)
    actor = BOWActor([attention], env.n_actions)
    run_gridworld(env, actor)


def test2():
    print('\n===\n=== test2: p_step_success=0.8 and per_step_cost=0.1\n===')
    env = gw.make_default_gridworld(per_step_cost=0.1, p_step_success=0.8)
    features = gw.GlobalGridFeatures(env.settings.width, env.settings.height)
    attention = AttendAt(features, position=lambda _: 0)
    actor = BOWActor([attention], env.n_actions)
    run_gridworld(env, actor)


def test3():
    print('\n===\n=== test3: p_step_success=0.8, but local features only\n===')
    env = gw.make_default_gridworld(p_step_success=0.8, start_random=True)
    features = gw.LocalGridFeatures()
    attention = AttendAt(features, position=lambda _: 0)
    actor = BOWActor([attention], env.n_actions)
    run_gridworld(env, actor)


def test4():
    print('\n===\n=== test4: big grid world, global features\n===')
    env = gw.make_big_gridworld()
    features = gw.GlobalGridFeatures(env.settings.width, env.settings.height)
    attention = AttendAt(features, position=lambda _: 0)
    actor = BOWActor([attention], env.n_actions)
    run_gridworld(env, actor)


if __name__ == '__main__':
    test0()
    test1()
    test2()
    test3()
    test4()
