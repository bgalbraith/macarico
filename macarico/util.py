from __future__ import division
import random
import sys
import itertools
from copy import deepcopy
import macarico
import numpy as np
import dynet as dy

from macarico.lts.lols import EpisodeRunner, one_step_deviation

# helpful functions

def reseed(seed=90210):
    random.seed(seed)
    #torch.manual_seed(seed)
    np.random.seed(seed)
    #dyparams = _dynet.DynetParams()
    #dyparams.from_args() # set some parameters manually
    #dyparams.init() # or init_from_params(dyparams)

def break_ties_by_policy(reference, policy, state, force_advance_policy=True):
    costs = np.zeros(state.n_actions)
    try:
        reference.set_min_costs_to_go(state, costs)
    except NotImplementedError:
        ref = reference(state)
        if force_advance_policy:
            policy(state)
        return ref
    # otherwise we successfully got costs
    old_actions = state.actions
    min_cost = min((costs[a] for a in old_actions))
    state.actions = [a for a in old_actions if costs[a] <= min_cost]
    a = policy(state)  # advances policy
    #print costs, old_actions, state.actions, a
    #a = state.actions[0]
    assert a is not None, 'got action None in %s, costs=%s, old_actions=%s' % (state.actions, costs, old_actions)
    state.actions = old_actions
    return a


def evaluate(data, policy, losses, verbose=False):
    "Compute average `loss()` of `policy` on `data`"
    was_list = True
    if not isinstance(losses, list):
        losses = [losses]
        was_list = False
    for loss in losses:
        loss.reset()
    for example in data:
        dy.renew_cg()
        env = example.mk_env()
        res = env.run_episode(policy)
        if verbose:
            print res, example
        for loss in losses:
            loss(example, env)
        dy.renew_cg()
    scores = [loss.get() for loss in losses]
    if not was_list:
        scores = scores[0]
    return scores


def should_print(print_freq, last_print, N):
    if print_freq is None:
        return False
    if last_print is None:
        return True
    next_print = last_print + print_freq if isinstance(print_freq, int) else \
                 last_print * print_freq
    return N >= next_print


def minibatch(data, minibatch_size, reshuffle):
    """
    >>> list(minibatch(range(8), 3, 0))
    [[0, 1, 2], [3, 4, 5], [6, 7]]

    >>> list(minibatch(range(0), 3, 0))
    []
    """
    # TODO this can prob be made way more efficient
    if reshuffle:
        random.shuffle(data)
    mb = []
    data = iter(data)
    try:
        prev_x = data.next()
    except StopIteration:
        # there are no examples
        return
    while True:
        mb.append(prev_x)
        try:
            prev_x = data.next()
        except StopIteration:
            break
        if len(mb) >= minibatch_size:
            yield mb, False
            mb = []
    if len(mb) > 0:
        yield mb, True

def padto(s, l):
    if isinstance(s, list):
        s = ' '.join(map(str, s))
    elif not isinstance(s, str):
        s = str(s)
    n = len(s)
    if n > l:
        return s[:l-2] + '..'
    return s + (' ' * (l - n))

def learner_to_alg(Learner, loss):
    def learning_alg(ex):
        env = ex.mk_env()
        learner = Learner()
        while True:
            env.run_episode(learner)
            if not hasattr(learner, 'run_again') or not learner.run_again():
                break
            env.rewind()
        loss_val = loss.evaluate(ex, env)
        learner.update(loss_val)
        return loss_val, getattr(learner, 'squared_loss', 0)
    return learning_alg


def learner_to_alg_ppo(Learner, loss):
    def learning_alg(ex):
        dy.renew_cg()
        env = ex.mk_env()
        learner = Learner()
        env.run_episode(learner)
        loss_val = loss.evaluate(ex, env)
        sq_loss = getattr(learner, 'squared_loss', 0)
        return loss_val, sq_loss, learner
    return learning_alg


def trainloop_ppo(training_data,
              n_actors=1,
              m_batch=1,
              k_epochs=1,
              dev_data=None,
              policy=None,
              Learner=None,
              learning_alg=None,
              optimizer=None,
              losses=None,      # one or more losses, first is used for early stopping
              run_per_batch=[],
              run_per_epoch=[],
              print_freq=2.0,   # int=additive, float=multiplicative
              quiet=False,
              train_eval_skip=100,
              reshuffle=True,
              print_dots=True,
              returned_parameters='best',  # { best, last, none }
              save_best_model_to=None,
              hogwild_rank=None,
              bandit_evaluation=False,
              dy_model=None,
              extra_dev_data=None,
              n_epochs=1,
             ):
    # n_epochs is always 1 for trainloop_ppo, we use n_actors, m_batches, k_epochs
    # to be consistent with the PPO paper
    assert(n_epochs == 1)
    assert(n_actors <= m_batches)
    if save_best_model_to is not None:
        assert dy_model is not None, \
            'if you want to save a model, you need to provide the dy.ParameterCollection as dy_model argument'

    assert (Learner is None) != (learning_alg is None), \
        'trainloop expects exactly one of Learner / learning_alg'

    assert losses is not None, \
        'must specify at least one loss function'


    if not isinstance(losses, list):
        losses = [losses]

    if learning_alg is None:
        learning_alg = learner_to_alg_ppo(Learner, losses[0])

    extra_loss_format = ''
    if not quiet:
        extra_loss_header = ''
        if len(losses) > 1:
            extra_loss_header += ' ' * 9
        for evaluator in losses[1:]:
            extra_loss_header += padto('  tr_' + evaluator.name, 10)
            extra_loss_header += padto('  de_' + evaluator.name, 10)
            extra_loss_format += '  %-8.5f  %-8.5f'
        if extra_dev_data is not None:
            extra_loss_header += '          |'
            extra_loss_format += ' |'
            for evaluator in losses:
                extra_loss_header += padto(' xd_' + evaluator.name, 10)
                extra_loss_format += ' %-8.5f'
        print >>sys.stderr, '%s | %s %s %8s  %5s  rand_dev_truth          rand_dev_pred%s' % \
            (padto('sq_err', 10),
             'tr_' + padto(losses[0].name, 8),
             'de_' + padto(losses[0].name, 8),
             'N', 'epoch', extra_loss_header)

    last_print = None
    best_de_err = float('inf')
    final_parameters = None
    error_history = []
    bandit_loss, bandit_count = 0., 0.

    if hogwild_rank is not None:
        reseed(20009 + 4837 * hogwild_rank)

    squared_loss, squared_loss_cnt = 0., 0.

    not_streaming = isinstance(training_data, list)

    N = 0  # total number of examples seen

    M = 0  # total number of examples seen this epoch

    # Convert train_data to batches of size N

    reshuffle = False
    # TODO: minibatching is really only useful if we can
    # preprocess in a useful way
    for batch, is_last_batch in minibatch(training_data, n_actors, reshuffle):
        learners = []
        for idx, ex in enumerate(batch):
            N += 1
            M += 1
            bl, sq, learner = learning_alg(ex)
            bandit_loss += bl
            bandit_count += 1
            squared_loss += sq
            squared_loss_cnt += 1
            if print_dots and not_streaming and (len(training_data) <= 40 or M % (len(training_data)//40) == 0):
                sys.stderr.write('.')

        if optimizer is not None:
            for k in range(k_epochs):
                for learner_batch, losses_batch in zip(learners_batches, losses_batches):
                    for learner_k, loss_k in zip(learner_batch, losses_batch):
                        dy.renew_cg()
                        learner_k.update(loss_k)
                optimizer.update()

        if should_print(print_freq, last_print, N) or is_last_batch:
            tr_err = [0] * len(losses)
            if bandit_evaluation:
                tr_err[0] = bandit_loss/bandit_count
            elif train_eval_skip is not None:
                tr_err = evaluate(training_data[::train_eval_skip], policy, losses)
            de_err = [0] * len(losses) if dev_data is None else \
                        evaluate(dev_data, policy, losses)

            ex_err = [] if extra_dev_data is None else evaluate(extra_dev_data, policy, losses)

            if not isinstance(tr_err, list): tr_err = [tr_err]
            if not isinstance(de_err, list): de_err = [de_err]
            if not isinstance(ex_err, list): de_err = [ex_err]

            extra_loss_scores = list(itertools.chain(*zip(tr_err[1:], de_err[1:])))
            if extra_dev_data is None:
                error_history.append((tr_err, de_err))
            else:
                error_history.append((tr_err, de_err, ex_err))

            random_dev_truth, random_dev_pred = '', ''
            if dev_data is not None:
                ex = random.choice(dev_data)
                random_dev_truth = ex
                random_dev_pred  = ex.mk_env().run_episode(policy)

            if not quiet and print_dots:
                sys.stderr.write('\r')

            fmt = '%-10.6f | %-10.6f  %-10.6f  %8s  %5s  [%s]  [%s]' + extra_loss_format
            is_best = de_err[0] < best_de_err
            if is_best:
                fmt += '  *'
            fmt_vals = [squared_loss / max(1, squared_loss_cnt),
                        tr_err[0],
                        de_err[0], N, 1,
                        padto(random_dev_truth, 20), padto(random_dev_pred, 20)] + \
                        extra_loss_scores + \
                        ex_err
            #print >>sys.stderr, '%g |' % (squared_loss / squared_loss_cnt),
            if not quiet:
                print >>sys.stderr, fmt % tuple(fmt_vals)

            last_print = N
            if is_best:
                best_de_err = de_err[0]
                if save_best_model_to is not None:
                    if print_dots and not quiet:
                        print >>sys.stderr, 'saving model to %s...' % save_best_model_to,
                    #torch.save(policy.state_dict(), save_best_model_to)
                    dy_model.save(save_best_model_to)
                    if print_dots and not quiet:
                        sys.stderr.write('\r' + (' ' * (21 + len(save_best_model_to))) + '\r')
                if returned_parameters == 'best':
                    final_parameters = None # deepcopy(policy)

        # TODO make sure run_per_batch is doing the correct thing
        for x in run_per_batch: x()
    for x in run_per_epoch: x()

    if returned_parameters == 'last':
        final_parameters = None # deepcopy(policy)

    return error_history, final_parameters

########################################################
def trainloop(training_data,
              dev_data=None,
              policy=None,
              Learner=None,
              learning_alg=None,
              optimizer=None,
              losses=None,      # one or more losses, first is used for early stopping
              n_epochs=10,
              minibatch_size=1,
              run_per_batch=[],
              run_per_epoch=[],
              print_freq=2.0,   # int=additive, float=multiplicative
              print_per_epoch=True,
              quiet=False,
              train_eval_skip=100,
              reshuffle=True,
              print_dots=True,
              returned_parameters='best',  # { best, last, none }
              save_best_model_to=None,
              hogwild_rank=None,
              bandit_evaluation=False,
              dy_model=None,
              extra_dev_data=None,
             ):
    if save_best_model_to is not None:
        assert dy_model is not None, \
            'if you want to save a model, you need to provide the dy.ParameterCollection as dy_model argument'

    assert (Learner is None) != (learning_alg is None), \
        'trainloop expects exactly one of Learner / learning_alg'

    assert losses is not None, \
        'must specify at least one loss function'

    if bandit_evaluation and n_epochs > 1 and not quiet:
        print >>sys.stderr, 'warning: running bandit mode with n_epochs>1, this is weird!'

    if not isinstance(losses, list):
        losses = [losses]

    if learning_alg is None:
        learning_alg = learner_to_alg(Learner, losses[0])

    extra_loss_format = ''
    if not quiet:
        extra_loss_header = ''
        if len(losses) > 1:
            extra_loss_header += ' ' * 9
        for evaluator in losses[1:]:
            extra_loss_header += padto('  tr_' + evaluator.name, 10)
            extra_loss_header += padto('  de_' + evaluator.name, 10)
            extra_loss_format += '  %-8.5f  %-8.5f'
        if extra_dev_data is not None:
            extra_loss_header += '          |'
            extra_loss_format += ' |'
            for evaluator in losses:
                extra_loss_header += padto(' xd_' + evaluator.name, 10)
                extra_loss_format += ' %-8.5f'
        print >>sys.stderr, '%s | %s %s %8s  %5s  rand_dev_truth          rand_dev_pred%s' % \
            (padto('sq_err', 10),
             'tr_' + padto(losses[0].name, 8),
             'de_' + padto(losses[0].name, 8),
             'N', 'epoch', extra_loss_header)

    last_print = None
    best_de_err = float('inf')
    final_parameters = None
    error_history = []
    bandit_loss, bandit_count = 0., 0.

    if hogwild_rank is not None:
        reseed(20009 + 4837 * hogwild_rank)

    squared_loss, squared_loss_cnt = 0., 0.

    not_streaming = isinstance(training_data, list)

    N = 0  # total number of examples seen
    for epoch in xrange(1, n_epochs+1):
        M = 0  # total number of examples seen this epoch
        for batch, is_last_batch in minibatch(training_data, minibatch_size, reshuffle):
            #if optimizer is not None:
                #optimizer.zero_grad()
            dy.renew_cg()
            # TODO: minibatching is really only useful if we can
            # preprocess in a useful way
            for ex in batch:
                N += 1
                M += 1
                bl, sq = learning_alg(ex)
                bandit_loss += bl
                bandit_count += 1
                squared_loss += sq
                squared_loss_cnt += 1
                if print_dots and not_streaming and (len(training_data) <= 40 or M % (len(training_data)//40) == 0):
                    sys.stderr.write('.')

            if optimizer is not None:
                optimizer.update()

            if should_print(print_freq, last_print, N) or \
               (is_last_batch and (print_per_epoch or (epoch==n_epochs))):
                tr_err = [0] * len(losses)
                if bandit_evaluation:
                    tr_err[0] = bandit_loss/bandit_count
                elif train_eval_skip is not None:
                    tr_err = evaluate(training_data[::train_eval_skip], policy, losses)
                de_err = [0] * len(losses) if dev_data is None else \
                         evaluate(dev_data, policy, losses)

                ex_err = [] if extra_dev_data is None else evaluate(extra_dev_data, policy, losses)

                if not isinstance(tr_err, list): tr_err = [tr_err]
                if not isinstance(de_err, list): de_err = [de_err]
                if not isinstance(ex_err, list): de_err = [ex_err]

                extra_loss_scores = list(itertools.chain(*zip(tr_err[1:], de_err[1:])))
                if extra_dev_data is None:
                    error_history.append((tr_err, de_err))
                else:
                    error_history.append((tr_err, de_err, ex_err))

                random_dev_truth, random_dev_pred = '', ''
                if dev_data is not None:
                    ex = random.choice(dev_data)
                    random_dev_truth = ex
                    random_dev_pred  = ex.mk_env().run_episode(policy)

                if not quiet and print_dots:
                    sys.stderr.write('\r')

                fmt = '%-10.6f | %-10.6f  %-10.6f  %8s  %5s  [%s]  [%s]' + extra_loss_format
                is_best = de_err[0] < best_de_err
                if is_best:
                    fmt += '  *'
                fmt_vals = [squared_loss / max(1, squared_loss_cnt),
                            tr_err[0],
                            de_err[0], N, epoch,
                            padto(random_dev_truth, 20), padto(random_dev_pred, 20)] + \
                           extra_loss_scores + \
                           ex_err
                #print >>sys.stderr, '%g |' % (squared_loss / squared_loss_cnt),
                if not quiet:
                    print >>sys.stderr, fmt % tuple(fmt_vals)

                last_print = N
                if is_best:
                    best_de_err = de_err[0]
                    if save_best_model_to is not None:
                        if print_dots and not quiet:
                            print >>sys.stderr, 'saving model to %s...' % save_best_model_to,
                        #torch.save(policy.state_dict(), save_best_model_to)
                        dy_model.save(save_best_model_to)
                        if print_dots and not quiet:
                            sys.stderr.write('\r' + (' ' * (21 + len(save_best_model_to))) + '\r')
                    if returned_parameters == 'best':
                        final_parameters = None # deepcopy(policy)

            for x in run_per_batch: x()
        for x in run_per_epoch: x()

    if returned_parameters == 'last':
        final_parameters = None # deepcopy(policy)

    return error_history, final_parameters

########################################################
# synthetic data construction

def make_sequence_reversal_data(num_ex, ex_len, n_types):
    data = []
    for _ in xrange(num_ex):
        x = [random.choice(range(n_types)) for _ in xrange(ex_len)]
        y = list(reversed(x))
        data.append((x,y))
    return data

def make_sequence_mod_data(num_ex, ex_len, n_types, n_labels):
    data = []
    for _ in xrange(num_ex):
        x = np.random.randint(n_types, size=ex_len)
        y = (x+1) % n_labels
        data.append((x,y))
    return data

def test_reference_on(ref, loss, ex, verbose=True, test_values=False, except_on_failure=True):
    from macarico import Policy
    from macarico.policies.linear import LinearPolicy

    env = ex.mk_env()
    policy = LinearPolicy(dy.ParameterCollection(), None, env.n_actions)

    def run(run_strategy):
        env.rewind()
        runner = EpisodeRunner(policy, run_strategy, ref)
        env.run_episode(runner)
        cost = loss()(ex, env)
        return cost, runner.trajectory, runner.limited_actions, runner.costs, runner.ref_costs

    # generate the backbone by REF
    loss0, traj0, limit0, costs0, refcosts0 = run(lambda t: EpisodeRunner.REF)
    if verbose:
        print 'loss0', loss0, 'traj0', traj0

    backbone = lambda t: (EpisodeRunner.ACT, traj0[t])
    n_actions = env.n_actions
    any_fail = False
    for t in xrange(len(traj0)):
        costs = np.zeros(n_actions)
        traj1_all = [None] * n_actions
        for a in limit0[t]:
            #if a == traj0[t]: continue
            l, traj1, _, _, _ = run(one_step_deviation(len(traj0), backbone, lambda _: EpisodeRunner.REF, t, a))
            #if verbose:
            #    print t, a, l
            costs[a] = l
            traj1_all[a] = traj1
            if l < loss0 or (a == traj0[t] and l != loss0):
                print 'local opt failure, ref loss=%g, loss=%g on deviation (%d, %d), traj0=%s traj\'=%s [ontraj=%s, is_proj=%s]' % \
                    (loss0, l, t, a, traj0, traj1, a == traj0[t], not ex.is_non_projective)
                any_fail = True
                if except_on_failure:
                    raise Exception()
        if test_values:
            for a in limit0[t]:
                if refcosts0[t][a] != costs[a]:
                    print 'cost failure, t=%d, a=%d, traj0=%s, traj1=%s, ref_costs=%s, observed costs=%s [is_proj=%s]' % \
                        (t, a, traj0, traj1_all[a], \
                         [refcosts0[t][a0] for a0 in limit0[t]], \
                         [costs[a0] for a0 in limit0[t]], \
                         not ex.is_non_projective)
                    if except_on_failure:
                        raise Exception()

    if not any_fail:
        print 'passed!'

def test_reference(ref, loss, data, verbose=False, test_values=False, except_on_failure=True):
    for n, ex in enumerate(data):
        print '# example %d ' % n,
        test_reference_on(ref, loss, ex, verbose, test_values, except_on_failure)

def sample_action_from_probs(r, np_probs):
    r0 = r
    for i, v in enumerate(np_probs):
        r -= v
        if r <= 0:
            return i
    mx = np.argmax(np_probs)
    print >>sys.stderr, 'warning: sampling from %s failed! returning max item %d; (r=%g r0=%g sum=%g)' % \
        (str(np_probs), mx, r, r0, np_probs.sum())
    return len(np_probs)-1

def sample_from_np_probs(np_probs):
    r = np.random.rand()
    a = sample_action_from_probs(r, np_probs)
    return a, np_probs[a]

def sample_from_probs(probs):
    r = np.random.rand()
    a = sample_action_from_probs(r, probs.npvalue())
    return a, probs[a]
