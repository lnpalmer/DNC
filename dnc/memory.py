import torch
import torch.nn as nn

class DNCMemory(nn.Module):
  """
  The memory component of a Diffentiable Neural Computer.
  """

  def __init__(self, entry_size):
    super().__init__()

    self.entry_size_ = entry_size
  
  def forward(self, interface, state):
  """
  Receives an interface vector and outputs memory reads.

  Arguments:
    interface (Tensor): the aggregated parameters for interaction with memory.
    state (tuple of Tensor's): the current state of the memory bank.
  """
  batch_size, _, num_entries = interface.size()

  memory, temporal_links = state

  def make_state(self, num_entries):
    """
    Makes an initial state for the memory.
    
    Arguments:
      num_entries (int): The number of discrete entries in the memory.
      batch_size (int): The batch size of the desired state.
    
    Returns:
      state:
        The prepared state.
    """
    memory = torch.zeros(batch_size, num_entries, self.entry_size_)
    temporal_links = torch.zeros(batch_size, num_entries, num_entries)

    return memory, temporal_links