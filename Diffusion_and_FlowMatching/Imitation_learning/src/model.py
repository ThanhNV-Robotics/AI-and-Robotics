"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy): # subclass or inherited class of BasePolicy
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int, # dimension of the input state
        action_dim: int, # dimension of each action
        chunk_size: int, # number of actions to predict in each chunk
        hidden_dims: tuple[int, ...] = (128, 128), # size of the hidden layers (default: 2 layers of 128 units each)
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size) # call the constructor of the BasePolicy to initialize state_dim, action_dim, and chunk_size
    # Build Multi-Layer Perceptron (MLP) for the policy network
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim)) # add a linear layer from input_dim to hidden_dim
            layers.append(nn.ReLU()) # add a ReLU activation function
            input_dim = hidden_dim # update input_dim for the next layer
        layers.append(nn.Linear(input_dim, chunk_size * action_dim)) # add a final linear layer to output the predicted action chunk
        self.mlp = nn.Sequential(*layers) # create the MLP as a sequential model
    
    # Implemnt MSE (Mean Squared Error) loss computation for training the MSEPolicy
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor: # return type
    # state: input state tensor with shape (batch, state_dim)
    # action_chunk: reference/ground-truth action chunk tensor with shape (batch, chunk_size, action_dim)
        
        # Compute forward pass
        pred_action_chunk = self.mlp(state)
        # compute MSE loss
        pred_action_chunk = pred_action_chunk.view(-1, self.chunk_size, self.action_dim) # reshape the predicted action chunk to (batch, chunk_size, action_dim)
        loss = nn.MSELoss()(pred_action_chunk, action_chunk) # compute the MSE loss between the predicted action chunk and the reference action chunk
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
    # this function generates a chunk of actions given an input state. For the MSEPolicy, we can simply do a forward pass through the MLP to get the predicted action chunk.
        with torch.no_grad(): # disable gradient computation since we are just sampling actions
            pred_action_chunk = self.mlp(state) # get the predicted action chunk from the MLP
            pred_action_chunk = pred_action_chunk.view(-1, self.chunk_size, self.action_dim) # reshape the predicted action chunk to (batch, chunk_size, action_dim)
        return pred_action_chunk
        # raise NotImplementedError


class FlowMatchingPolicy(BasePolicy): # This is for hw1 part 3: Action Chunking with Flow Matching
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # Build Multi-Layer Perceptron (MLP) for the velocity network v_θ
        # Input: state + flattened noisy action + timestep τ
        # Output: velocity (same shape as action chunk when reshaped)
        layers = []
        input_dim = state_dim + chunk_size * action_dim + 1  # state + noisy_action + timestep
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            state: (batch, state_dim)
            action_chunk: (batch, chunk_size, action_dim) - the target action chunk A_t
        
        Returns:
            Flow matching loss (scalar)
        """
        batch_size = state.shape[0]
        
        # Sample flow matching timestep τ ~ U(0,1) for each sample in batch
        tau = torch.rand(batch_size, 1, device=state.device)  # (batch, 1)
        
        # Sample noise A_{t,0} ~ N(0,I) with same shape as action_chunk
        noise = torch.randn_like(action_chunk)  # (batch, chunk_size, action_dim)
        
        # Compute interpolation: A_{t,τ} = τ*A_t + (1-τ)*A_{t,0}
        noisy_action = tau.unsqueeze(-1) * action_chunk + (1 - tau.unsqueeze(-1)) * noise
        
        # Flatten noisy action for network input
        noisy_action_flat = noisy_action.view(batch_size, -1)  # (batch, chunk_size * action_dim)
        
        # Concatenate state, noisy action, and timestep
        network_input = torch.cat([state, noisy_action_flat, tau], dim=1)
        
        # Predict velocity v_θ(o_t, A_{t,τ}, τ)
        pred_velocity = self.mlp(network_input)  # (batch, chunk_size * action_dim)
        pred_velocity = pred_velocity.view(batch_size, self.chunk_size, self.action_dim)
        
        # Target velocity: A_t - A_{t,0} (the direction from noise to clean action)
        target_velocity = action_chunk - noise
        
        # Compute MSE loss: ||v_θ - (A_t - A_{t,0})||_2^2
        loss = nn.MSELoss()(pred_velocity, target_velocity)
        
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate action chunks using Euler integration.
        
        Args:
            state: (batch, state_dim)
            num_steps: number of integration steps (n in the paper)
        
        Returns:
            action_chunk: (batch, chunk_size, action_dim)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample initial noise A_{t,0} ~ N(0,I)
        action = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)
        
        # Euler integration from τ=0 to τ=1 with n steps
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for step in range(num_steps):
                # Current timestep τ
                tau = torch.full((batch_size, 1), step * dt, device=device)
                
                # Flatten action for network input
                action_flat = action.view(batch_size, -1) # reshape to 2D vector (batch_size, chunk_size*action_dim)
                
                # Concatenate state, current action, and timestep
                network_input = torch.cat([state, action_flat, tau], dim=1)
                
                # Predict velocity v_θ(o_t, A_{t,τ}, τ)
                velocity = self.mlp(network_input)
                velocity = velocity.view(batch_size, self.chunk_size, self.action_dim)
                
                # Euler update: A_{t,τ+dt} = A_{t,τ} + dt * v_θ
                action = action + dt * velocity
        
        return action


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128), # default config
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

# build a test main function here
if __name__ == "__main__":
    # Example usage of build_policy
    # in the description:
    # observation dim = 5: position and orientation of the T block (x,y,theta), position of the agent (x,y)
    # action dim = 2: position of the agent (x,y)
    state_dim = 5
    action_dim = 2
    chunk_size = 8
    hidden_dims = (128, 128) # 2 hidden layers with 128 units each

    # create an MSE policy
    mse_policy = build_policy(
        policy_type="mse",
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden_dims=hidden_dims,
    )

    # print some information about the policy
    print("MSE Policy:")
    print(f"State dim: {mse_policy.state_dim}")
    print(f"Action dim: {mse_policy.action_dim}")
    print(f"Chunk size: {mse_policy.chunk_size}") # Prediction horizon of the action