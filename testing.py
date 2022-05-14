# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:16:33 2022

@author: asus
"""
import pytest

from enum import Enum
import numpy as np




state = np.array([1, 0, 0, 4])

index_n_active_genes = 0
index_n_inactive_genes = 1
index_n_RNAs = 2
index_n_proteins = 3

class Transition(Enum):
    GENE_ACTIVATE = 'gene activate'
    GENE_INACTIVATE = 'gene inactivate'
    RNA_INCREASE = 'RNA increase'
    RNA_DEGRADE = 'RNA degrade'
    PROTEIN_INCREASE = 'Protein increase'
    PROTEIN_DEGRADE = 'Protein degrade'

def update_state(event, state):
    """This method updates the initial state according to the event occured
    
    Parameters
    ----------
    event : member of Transition class
    state : ndarray
            ndarray with 4 dimensions: number of active genes, number of
            inactive genes, number of RNAs and number of proteins

    Returns
    -------
    updated_state : ndarray
                    ndarray with 4 dimensions: number of active genes, number of
                    inactive genes, number of RNAs and number of proteins after
                    the event has occured.
    """

    
    if event == Transition.GENE_ACTIVATE:
         state[index_n_active_genes] +=1 
         state[index_n_inactive_genes] -=1
         state = state
         state = state.copy()

         
    elif event == Transition.GENE_INACTIVATE:
        state[index_n_active_genes] -=1
        state[index_n_inactive_genes] +=1
        state = state
        state = state.copy()

    elif event == Transition.RNA_INCREASE:
         state[index_n_RNAs] +=1
         state = state
         state = state.copy()

    elif event == Transition.RNA_DEGRADE:
        state[index_n_RNAs] -=1
        state = state
        state = state.copy()

    elif event == Transition.PROTEIN_INCREASE:
         state[index_n_proteins] +=1
         state = state
         state = state.copy()

    elif event == Transition.PROTEIN_DEGRADE:
        state[index_n_proteins] -=1
        state = state
        state = state.copy()



        
    elif isinstance(event,str) or isinstance(event, str):
        raise TypeError("Do not use string ! Choose transitions from Transition enum members.")
    else:
        raise ValueError("Transition not recognized")
    

    
    updated_state = state
    return updated_state 




#%% Tests

def test_update_state_return_correct_gene_state_when_gene_is_active():
    """Verify that the update_state function updates gene state to 'inactive' 
    when the initial gene state is active and there is a transition that
    inactivates gene and verify that in this transition the number of RNAs 
    and proteins do not change"""
    state = np.array([1, 0, 0, 0])
    event = Transition.GENE_INACTIVATE
    updated_state = update_state(event, state) 
    assert updated_state.any() == np.array([0,1,0,0]).any()

def test_update_state_return_correct_gene_state_when_gene_is_inactive():
    """Verify that the update_state function updates gene state to 'active' 
    when the initial gene state is inactive and when there is a transition
    that activates gene and verify that in this transition the number of RNAs 
    and proteins do not change
    """
    state = np.array([0, 1, 0, 0])
    event = Transition.GENE_ACTIVATE
    updated_state = update_state(event, state) 
    assert (updated_state == np.array([1,0,0,0])).all()

def test_update_state_return_correct_value_when_gene_is_active_and_RNA_INCREASE():
    """Verify that the update_state function increases the number of RNA
    molecules by 1 when gene state is active and the active gene state transition
    is Transition.RNA_INCREASE.
    """
    state = np.array([1, 0, 0, 0]) 
    event = Transition.RNA_INCREASE
    updated_state = update_state(event, state) 
    assert (updated_state == np.array([1,0,1,0])).all()

def test_update_state_return_correct_value_when_gene_is_active_and_RNA_DECREASE():
    state = np.array([1, 0, 0, 0]) 
    event = Transition.RNA_DEGRADE
    updated_state = update_state(event, state) 
    assert (updated_state == np.array([1,0,-1,0])).all()

def test_update_state_return_correct_value_when_gene_is_active_and_Protein_DEGRADE():
    state = np.array([1, 0, 1, 1])
    event = Transition.PROTEIN_DEGRADE
    updated_state = update_state(event, state) 
    assert (updated_state == np.array([1,0,1,0])).all()


