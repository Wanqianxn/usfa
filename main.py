
import torch
import os, time
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import torch.distributions as ds


GAMMA = 0.99 # Discount rate
ALPHA = 0.1 # Update rate
EPSILON = 0.1 # Exploration factor (epsilon-greedy)
BETA = 10 # Softmax inverse temperature, set to e.g. 50 for max instead of softmax


class DQN(torch.nn.Module):
    """ Who cares about Bellman equations when you have *deep* neural networks? """

    def __init__(self, isize, osize, hsize=50, zerofy=True):
        """ Instantiates object, defines network architecture. """
        super(DQN, self).__init__()
        self.L1 = torch.nn.Linear(isize, hsize)
        self.L2 = torch.nn.Linear(hsize, osize)
        self.relu = torch.nn.ReLU()
        if zerofy:
            torch.nn.init.constant_(self.L1.weight, 0.)
            torch.nn.init.constant_(self.L1.bias, 0.)
            torch.nn.init.constant_(self.L2.weight, 0.)
            torch.nn.init.constant_(self.L2.bias, 0.)

    def forward(self, X):
        """ Computes the forward pass. """
        out = self.relu(self.L1(X))
        out = self.L2(out)
        return out


class MTRL:
    """ Multi-task RL on a toy problem.
        Function Approximator: DQN.
        Algorithms: Value Iteration, Q-Learning (Tabular + DQN), UVFA, SF, USFA.
    """
    def __init__(self, Wtrain, Wtest, load_index=1):
        """ Instantiates object, load dataset. """
        getattr(self, f'load_dataset_{load_index}')
        self.Wtrain = torch.tensor(Wtrain)
        self.Ntrain = len(self.Wtrain)
        self.Wtest = torch.tensor(Wtest)
        self.Ntest = len(self.Wtest)

    def load_dataset_1(self):
        """ Load MDP = (S, A, T, R) for Dataset 1. """

        self.NS = 13 # NS = |S|, number of states
        self.NA = 3 # NA = |A|, number of actions
        self.NP = 3 # NP = |phi| = |w|, dimensionality of task-space and feature-space
        self.S = torch.arange(self.NS) # S, set of states
        self.A = torch.arange(self.NA) # A, set of actions

        # T(s,a,s') -- state transition function. (NS, NA, NS) tensor.
        self.T = torch.zeros(self.NS, self.NA, self.NS)
        self.T[0, 0, 1] = 1.
        self.T[0, 1, 2] = 1.
        self.T[0, 2, 3] = 1.
        self.T[1, 0, 4] = 1.
        self.T[1, 1, 5] = 1.
        self.T[1, 2, 6] = 1.
        self.T[2, 0, 7] = 1.
        self.T[2, 1, 8] = 1.
        self.T[2, 2, 9] = 1.
        self.T[3, 0, 10] = 1.
        self.T[3, 1, 11] = 1.
        self.T[3, 2, 12] = 1.

        # Φ(s) -- state features. (NS, 3) tensor.
        self.phi = []
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0])
        self.phi.append([0, 0, 0.9])
        self.phi.append([1, 0, 0.9])
        self.phi.append([0, 0, 0.9])
        self.phi.append([0.9, 0, 0.9])
        self.phi.append([0.8, 0.8, 0.9])
        self.phi.append([0, 0.9, 0.9])
        self.phi.append([0, 0, 0.9])
        self.phi.append([0, 1, 1.5])
        self.phi.append([0, 0, 0.9])
        self.phi = torch.tensor(self.phi)

        # (NS, ) boolean tensor specifying if state is terminal.
        self.is_terminal = torch.tensor([0,0,0,0,1,1,1,1,1,1,1,1,1])

    def q_learning(self, npertask=200, gamma=GAMMA, alpha=ALPHA, eps=EPSILON, beta=BETA):
        """ Vanilla (tabular) Q-learning algorithm. """
        train_tasks = torch.arange(self.Ntrain).repeat(npertask)
        train_tasks = train_tasks[torch.randperm(len(train_tasks))]
        Q = torch.rand(self.NS, self.NA)
        for i in range(self.NS):
            if self.is_terminal[i]:
                Q[i] *= 0.

        for k in train_tasks:
            state = 0
            while not self.is_terminal[state]:
                actions = torch.randint(self.NA, (self.NS, ), dtype=torch.long)
                if stats.uniform.rvs() < 1. - eps:
                    actions = Q.argmax(dim=1)
                action = actions[state]
                new_state = ds.Categorical(self.T[state, action, :]).sample()
                reward = self.phi[new_state] @ self.Wtrain[k]
                new_action = actions[new_state]
                Q[state, action] += alpha * (reward + gamma * Q[new_state, new_action] - Q[state, action])
                state = new_state
                print(Q)

        pi = torch.nn.Softmax(dim=1)(beta * Q)
        return Q, pi

    def dqn_learning(self, npertask=200, gamma=GAMMA, alpha=ALPHA, eps=EPSILON, beta=BETA):
        """ Vanilla (tabular) Q-learning algorithm. """
        def compute_all(dqn_model):
            ip1 = encS.repeat(1, self.NA).view(-1, self.NS)
            ip2 = encA.repeat(self.NS, 1)
            out = dqn_model(torch.cat((ip1, ip2), 1)).squeeze().view(self.NS, self.NA)
            return out

        train_tasks = torch.arange(self.Ntrain).repeat(npertask)
        train_tasks = train_tasks[torch.randperm(len(train_tasks))]
        Q = DQN(self.NS + self.NA, 1, zerofy=False)
        optim = torch.optim.Adagrad(Q.parameters(), lr=alpha)
        loss = torch.nn.MSELoss()
        encS = torch.eye(self.NS)
        encA = torch.eye(self.NA)

        for k in train_tasks:
            old_model = compute_all(Q)
            state = 0
            while not self.is_terminal[state]:
                optim.zero_grad()
                current = compute_all(Q)
                actions = torch.randint(self.NA, (self.NS, ), dtype=torch.long)
                if stats.uniform.rvs() < 1. - eps:
                    actions = current.argmax(dim=1)
                action = actions[state]
                current[state,action].backward()
                new_state = ds.Categorical(self.T[state, action, :]).sample()
                reward = (self.phi[new_state] @ self.Wtrain[k])
                #output = loss(current[state,action], reward + gamma * old_model[new_state].max(dim=0)[0])
                for mp in Q.parameters():
                    mp.grad *= alpha * (reward + gamma * old_model[new_state].max(dim=0)[0] - current[state,action])
                old_model = current.detach()
                optim.step()
                state = new_state
        pi = torch.nn.Softmax(dim=1)(beta * compute_all(Q))
        return Q, pi

    def value_iteration(self, w, threshold=0.01, gamma=GAMMA, beta=BETA):
        """ Value iteration algorithm. """
        V = torch.zeros(self.NS)
        pi = torch.zeros(self.NS, self.NA)
        for i in range(self.NS):
            if self.is_terminal[i]:
                V[i] =  self.phi[i] @ w

        delta = torch.tensor([threshold])
        count = 0
        while delta >= threshold:
            delta = torch.tensor([0.])
            count += 1
            for s in range(self.NS):
                curr_state = V[s].clone()
                vals = self.phi @ w
                vals += gamma * V
                Q = self.T[s] @ vals
                pi[s] = torch.nn.Softmax(dim=0)(beta * Q)
                V[s] = Q.max(dim=0)[0]
                delta = max(delta, abs(V[s] - curr_state))
        # print(f'VI converged in {count} iterations.')
        return V, pi

    def uvfa_train(self, npertask=200, epochs=10, bsize=10, gamma=GAMMA, alpha=ALPHA, beta=BETA):
        """ Universal value function approximators. """
        Vtrain = torch.zeros(self.Ntrain, self.NS)
        for widx in range(self.Ntrain):
            Vtrain[widx] = self.value_iteration(self.Wtrain[widx], gamma=gamma, beta=beta)[0]

        UVFN = DQN(self.NS + self.NP, 1)
        optim = torch.optim.Adagrad(UVFN.parameters(), lr=alpha)
        loss = torch.nn.MSELoss()

        train_tasks = torch.arange(self.Ntrain).repeat(npertask)
        X = self.Wtrain[train_tasks].repeat(1, self.NS).view(-1, self.NP)
        X = torch.cat((torch.eye(self.NS).repeat(npertask * self.Ntrain, 1), X), 1)
        Y = Vtrain.flatten().repeat(npertask).unsqueeze(dim=0).t()
        XY = torch.cat((X, Y), 1)
        XY = XY[torch.randperm(len(XY))]
        XY = XY.split(bsize)

        for epoch in range(epochs):
            epoch_loss = 0.
            for i in range(bsize):
                optim.zero_grad()
                preds = UVFN(XY[i][:,:-1])
                target = XY[i][:,-1]
                output = loss(preds.squeeze(), target)
                epoch_loss += output.item()
                output.backward()
                optim.step()
            print(f'Epoch {epoch}: {epoch_loss}')
        return UVFN

    def uvfa_predict(self, uvfn, w, gamma=GAMMA, beta=BETA):
        """ Returns optimal policy for a new task. """
        inp = torch.cat((torch.eye(self.NS), w.unsqueeze(dim=0).repeat(self.NS, 1)), 1)
        V = uvfn(inp).squeeze()
        pi = torch.zeros(self.NS, self.NA)
        for s in range(self.NS):
            vals = self.phi @ w
            vals += gamma * V
            Q = self.T[s] @ vals
            pi[s] = torch.nn.Softmax(dim=0)(beta * Q)
        return pi

    def sfgpi_train(self, threshold=0.01, gamma=GAMMA, beta=BETA):
        """ Successor features + general policy iteration. """
        VPtrain = []
        for wtr in self.Wtrain:
            VPtrain.append(self.value_iteration(wtr, gamma=gamma, beta=beta))

        psi = self.phi.unsqueeze(dim=0).repeat(self.Ntrain, 1, 1)
        for widx in range(self.Ntrain):
            delta = torch.tensor([threshold])
            pi = VPtrain[widx][1]
            count = 0
            while delta >= threshold:
                delta = torch.tensor([0.])
                count += 1
                for s in range(self.NS):
                    curr_psi = psi[widx,s].clone()
                    psi[widx,s] = self.phi[s] + gamma * (pi[s] @ self.T[s] @ psi[widx])
                    delta = max(delta, torch.dist(psi[widx,s], curr_psi))
            print(f'SF + GPI converged in {count} iterations.')
        return psi

    def sfgpi_predict(self, psi, w, gamma=GAMMA, beta=BETA):
        """ Returns optimal policy for a new task. """
        vmax = (psi @ w).max(dim=0)[0]
        Q = self.T @ ((self.phi @ w) + gamma * vmax)
        pi = torch.nn.Softmax(dim=1)(beta * Q)
        return pi

    def usfa_train(self, niters=200, nz=10, zsigma=0.3, threshold=0.01, alpha=ALPHA, gamma=GAMMA, beta=BETA):
        """ Universal successor feature approximators. """
        USFN = DQN(self.NS + self.NP, self.NP, zerofy=False)
        encS = torch.eye(self.NS)
        optim = torch.optim.Adagrad(USFN.parameters(), lr=alpha)
        loss = torch.nn.MSELoss()

        for niter in range(1, niters + 1):
            # Sample training task w ~ M.
            state = 0
            widx = np.random.randint(self.Ntrain)
            wtask = self.Wtrain[widx]

            # Sample policies z ~ D(·|w), and get V* and π* for each z.
            ztasks = ds.MultivariateNormal(wtask, zsigma * torch.eye(self.NP)).sample(torch.Size([nz]))
            zvalues = [self.value_iteration(zt, gamma=gamma, beta=beta) for zt in ztasks]

            # Use "policy evaluation" to get the correct Ψ(s,z) for each z. This is the "SF" part.
            psi = self.phi.unsqueeze(dim=0).repeat(nz, 1, 1) # shape: (nz, NS, 3)
            for zidx in range(nz):
                delta = torch.tensor([threshold])
                pi = zvalues[zidx][1]
                while delta >= threshold:
                    delta = torch.tensor([0.])
                    for s in range(self.NS):
                        curr_psi = psi[zidx,s].clone()
                        psi[zidx,s] = self.phi[s] + gamma * (pi[s] @ self.T[s] @ psi[zidx])
                        delta = max(delta, torch.dist(psi[zidx,s], curr_psi))

            # Use correct Ψ(s,z) to train Ψ^(s,z) neural network approximator. This is the "UVFA" part
            optim.zero_grad()
            ip1 = encS.repeat(nz, 1)
            ip2 = ztasks.repeat(1, self.NS).view(-1, self.NP)
            ip = torch.cat((ip1, ip2), 1)
            preds = USFN(ip) # shape: (NS * nz, 3)
            targets = psi.view(-1, self.NP)
            output = loss(preds, targets)
            if niter % 10 == 0:
                print(f'Epoch {niter}: {output}')
            output.backward()
            optim.step()
        return USFN

    def usfa_predict(self, usfn, w, C=None, beta=BETA, gamma=GAMMA):
        """ Returns optimal policy for a new task. """
        if type(C) == type(None):
            C = self.Wtrain
        NC = len(C)

        # Use learnt Ψ^(s,z) to get π(s). This is the "GPI" part.
        ip1 = torch.eye(self.NS).repeat(1, NC).view(-1, self.NS)
        ip2 = C.repeat(self.NS, 1)
        ip = torch.cat((ip1, ip2), 1)
        vals = (usfn(ip) @ w).split(NC)
        vmax = torch.tensor([vv.max(dim=0)[0] for vv in vals])

        Q = self.T @ ((self.phi @ w) + gamma * vmax)
        pi = torch.nn.Softmax(dim=1)(beta * Q)
        return pi


def example():
    """ A working example of an MTRL instance. """
    wtrain = [[1., 0, 0], [0., 1, 0]]
    wtest = [[1., 1, 0], [0., 0, 1]]
    model = MTRL(wtrain, wtest)
    model.load_dataset_1()

    # Q-learning / DQN
    if False:
        Q, pi = model.q_learning()
        for wt in model.Wtest:
            print(pi)
        print('Q-learning works.')

    if True:
        Q, pi = model.dqn_learning(alpha=100.0)
        for wt in model.Wtest:
            print(pi)
        print('DQN Q-learning works.')

    # Value iteration
    if False:
        for wt in model.Wtest:
            V, pi = model.value_iteration(wt)
            print(pi)
        print('Value iteration works.')

    # UVFA
    if False:
        uvfn = model.uvfa_train()
        for wt in model.Wtest:
            pi = model.uvfa_predict(uvfn, wt)
            print(pi)
        print('UVFA works.')

    # SF + GPI
    if False:
        psi = model.sfgpi_train()
        for wt in model.Wtest:
            pi = model.sfgpi_predict(psi, wt)
            print(pi)
        print('SF + GPI works.')

    # USFA
    if True:
        usfn = model.usfa_train(alpha=0.001)
        for wt in model.Wtest:
            pi = model.usfa_predict(usfn, wt, C=wt.unsqueeze(dim=0))
            print(pi)
        print('USFA works.')

example()
