import numpy as np
import torch
import torch.nn as nn

import macarico
from macarico.actors import BOWActor, RNNActor
from macarico.features.sequence import AttendAt
from macarico.lts.dagger import DAgger
from macarico.lts.reinforce import Reinforce
from macarico.policies.linear import CSOAAPolicy
from macarico.tasks.concentration import Concentration, ConcentrationLoss, \
    ConcentrationPOFeatures, ConcentrationSmartFeatures, ConcentrationReference


def test(teacher: bool = True, imitation: bool = True):
    n_card_types = 3
    n_epochs = 500000
    print(f'Concentration: n_card_types {n_card_types}, teacher {teacher}, imitation {imitation}')
    
    env = Concentration(n_card_types=n_card_types, random_deck_per_episode=True)

    if teacher:
        features = ConcentrationSmartFeatures(n_card_types, cheat=False)
        features = macarico.Torch(features,
                                  20,
                                  [nn.Linear(features.dim, 20),
                                   nn.ReLU(),
                                  ])
        attention = AttendAt(features, position=lambda _: 0)
        actor = BOWActor([attention], env.n_actions)

    else:  # student
        features = ConcentrationPOFeatures()
        attention = AttendAt(features, position=lambda _: 0)
        actor = RNNActor([attention], env.n_actions, d_hid=50)
        
    policy = CSOAAPolicy(actor, env.n_actions)
    reference = ConcentrationReference()

    learner = DAgger(policy, reference) if imitation else Reinforce(policy)
    loss_fn = ConcentrationLoss()

    print(learner)

    ref_losses = []
    for epoch in range(1000):
        env.run_episode(reference)
        ref_losses.append(loss_fn(env.example))
    print('average reference loss %g' % np.mean(ref_losses))
    
    rnd_losses = []
    for epoch in range(1000):
        env.run_episode(lambda s: np.random.choice(list(s.actions)))
        rnd_losses.append(loss_fn(env.example))
    print('average random loss %g' % np.mean(rnd_losses))

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
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
            if best_loss is None or loss < best_loss[0]: best_loss = (loss, epoch)
            print(epoch, 'losses', loss, 'objective', np.mean(objs[-500:]), 'best_loss', best_loss, 'init_losses', np.mean(losses[:1000]), sum(env.example.costs), env.card_seq)
            if loss <= 0.99 * np.mean(ref_losses):
                break


if __name__ == '__main__':
    for imitation in [True, False]:
        for teacher in [True, False]:
            test(teacher, imitation)
            print("\n=====================\n")
