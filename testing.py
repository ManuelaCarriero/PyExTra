# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:16:33 2022

@author: asus
"""
import pytest

from enum import Enum
import numpy as np


# Define transitions 

class Transition(Enum):
    GENE_ACTIVATE = 'gene activate'
    GENE_INACTIVATE = 'gene inactivate'
    RNA_INCREASE = 'RNA increase'
    RNA_DEGRADE = 'RNA degrade'
    PROTEIN_INCREASE = 'Protein increase'
    PROTEIN_DEGRADE = 'Protein degrade'

def update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state):
    """This method updates the initial state according to the event occured
    
    Parameters
    ----------
    active_gene_event : Transition class member
                        Transition that occurs when gene state is active.
    inactive_gene_event: Transition class member
                         Transition that occurs when gene state is inactive.
    gene_state: str
                gene state that can be active or inactive
    RNA_Protein_state: ndarray
                       ndarray that has number of RNAs as first dimension 
                       and number of proteins as
                       second dimension.

    Returns
    -------
    updated_state : ndarray, shape (gene_state, RNA_Protein_state)
                    ndarray that has the gene state as first dimension and 
                    RNA and Protein number of molecules as second dimension
                    after a given event is occured.
    """
    if inactive_gene_event == Transition.GENE_ACTIVATE and gene_state == 'inactive':
        gene_state = 'active'
        gene_state = gene_state[:]
        RNA_Protein_state = RNA_Protein_state
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif active_gene_event == Transition.RNA_INCREASE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[0] +=1
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif active_gene_event == Transition.RNA_DEGRADE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[0] -=1
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif active_gene_event == Transition.PROTEIN_INCREASE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] +=1
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif active_gene_event == Transition.PROTEIN_DEGRADE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] -=1
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif active_gene_event == Transition.GENE_INACTIVATE and gene_state == 'active':
        gene_state = 'inactive'
        gene_state = gene_state[:]
        RNA_Protein_state = RNA_Protein_state
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif inactive_gene_event == Transition.RNA_DEGRADE and gene_state == 'inactive':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[0] -=1
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif inactive_gene_event == Transition.PROTEIN_INCREASE and gene_state == 'inactive' :
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] +=1
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif inactive_gene_event == Transition.PROTEIN_DEGRADE and gene_state == 'inactive':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] -=1
        RNA_Protein_state = RNA_Protein_state.copy() 
        
    elif isinstance(inactive_gene_event,str) or isinstance(active_gene_event, str):
        raise TypeError("Do not use string ! Choose transitions from Transition enum members.")
    else:
        raise ValueError("Transition not recognized")
        
    updated_state = np.array([gene_state,RNA_Protein_state], dtype=object)
    
    return updated_state





#%% Tests
def test_update_state_return_correct_gene_state_when_gene_is_active():
    """Verify that the update_state function updates gene state to 'inactive' 
    when the initial gene state is active and when the transition for gene in active state
    is "Transition.GENE_INACTIVATE" despite gene inactive transition is Transition.GENE_ACTIVATE
    and verify that in this transition the number of RNAs and proteins keeps constant"""
    active_gene_event = Transition.GENE_INACTIVATE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'inactive'
    assert updated_state[1].any() == np.array([0,0]).any()

def test_update_state_return_correct_gene_state_when_gene_is_inactive():
    """Verify that the update_state function updates gene state to 'active' 
    when the initial gene state is inactive and when the transition for gene in inactive state
    is "Transition.GENE_ACTIVATE" despite gene active transition is Transition.GENE_INACTIVATE
    and verify that in this transition the number of RNAs and proteins keeps constant
    """
    active_gene_event = Transition.GENE_INACTIVATE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'inactive'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'active'
    assert updated_state[1].any() == np.array([0,0]).any()

def test_update_state_return_correct_value_when_gene_is_active_and_RNA_INCREASE():
    """Verify that the update_state function increases the number of RNA
    molecules by 1 when gene state is active and the active gene state transition
    is Transition.RNA_INCREASE despite gene inactive transition is Transition.GENE_ACTIVATE.
    """
    active_gene_event = Transition.RNA_INCREASE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'active'
    assert updated_state[1].any() == np.array([1,0]).any()
    
def test_update_state_return_correct_value_when_gene_is_active_and_RNA_DEGRADE():
    """Verify that the update_state function decreases the number of RNA
    molecules by 1 when gene state is active and the active gene state transition
    is Transition.RNA_DEGRADE, despite gene inactive transition is Transition.GENE_ACTIVATE.
    """
    active_gene_event = Transition.RNA_DEGRADE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'active'
    assert updated_state[1].any() == np.array([-1,0]).any()

def test_update_state_return_correct_value_when_gene_is_active_and_Protein_INCREASE():
    """Verify that the update_state function increases the number of proteins
    by 1 when gene state is active and the active gene state transition
    is Transition.PROTEIN_INCREASE, despite gene inactive transition is Transition.GENE_ACTIVATE.
    """
    active_gene_event = Transition.PROTEIN_INCREASE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'active'
    assert updated_state[1].any() == np.array([0,1]).any()

def test_update_state_return_correct_value_when_gene_is_active_and_Protein_DEGRADE():
    """Verify that the update_state function decreases the number of proteins
    by 1 when gene state is active and the active gene state transition
    is Transition.PROTEIN_DEGRADE, despite gene inactive transition is Transition.GENE_ACTIVATE.
    """
    active_gene_event = Transition.PROTEIN_DEGRADE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'active'
    assert updated_state[1].any() == np.array([0,-1]).any()

def test_update_state_return_correct_value_when_gene_is_inactive_and_RNA_DEGRADE():
    """Verify that the update_state function decreases the number of RNA
    molecules by 1 when gene state is inactive and the inactive gene state transition
    is Transition.RNA_DEGRADE despite gene active transition is Transition.RNA_INCREASE.
    """
    active_gene_event = Transition.RNA_INCREASE
    inactive_gene_event = Transition.RNA_DEGRADE
    gene_state = 'inactive'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'inactive'
    assert updated_state[1].any() == np.array([-1,0]).any()

def test_update_state_return_correct_value_when_gene_is_inactive_and_Protein_INCREASE():
    """Verify that the update_state function increases the number of RNA
    molecules by 1 when gene state is inactive and the inactive gene state transition
    is Transition.PROTEIN_INCREASE despite gene active transition is Transition.RNA_INCREASE.
    """
    active_gene_event = Transition.RNA_INCREASE
    inactive_gene_event = Transition.PROTEIN_INCREASE
    gene_state = 'inactive'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'inactive'
    assert updated_state[1].any() == np.array([0,1]).any()

def test_update_state_return_correct_value_when_gene_is_inactive_and_Protein_DEGRADE():
    """Verify that the update_state function decreases the number of proteins
    molecules by 1 when gene state is inactive and the inactive gene state transition
    is Transition.PROTEIN_DEGRADE despite gene active transition is Transition.RNA_INCREASE.
    """
    active_gene_event = Transition.RNA_INCREASE
    inactive_gene_event = Transition.PROTEIN_DEGRADE
    gene_state = 'inactive'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert updated_state[0] == 'inactive'
    assert updated_state[1].any() == np.array([0,-1]).any()

def test_update_state_return_correct_array_length():
    """Verify that the update_state function returns a ndarray
    with length = 2 (one dimension for gene state and the other
    for the number of RNAs and proteins)"""
    active_gene_event = Transition.GENE_INACTIVATE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert len(updated_state) == 2

def test_update_state_return_correct_RNA_Protein_array_length():
    """Verify that the update_state function returns a ndarray
    whose second element is a ndarray of length = 2 (one dimension for 
    for number of RNAs and one for number of proteins)"""
    active_gene_event = Transition.GENE_INACTIVATE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert len(updated_state[1]) == 2

def test_update_state_return_correct_gene_state_array_length():
    """Verify that the update_state function returns a ndarray
    whose first element is a string of length = 8 (i.e. length of string 'inactive')
    when the initial gene state is active and when the transition for gene in active state
    is "Transition.GENE_INACTIVATE"
    """
    active_gene_event = Transition.GENE_INACTIVATE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert len(updated_state[0]) == 8

def test_update_state_return_a_string_as_gene_state():
    """Verify that the update_state function returns a ndarray that has a string 
    as first element of ndarray when the initial gene state is active and 
    when the transition for gene in active state
    is "Transition.GENE_INACTIVATE"
    """
    active_gene_event = Transition.GENE_INACTIVATE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
    assert isinstance(updated_state[0], str)
    
def test_update_state_return_value_error_given_wrong_transition():
    """Verify that the update_state function raises a 'value' error
    given the wrong transition
    """
    active_gene_event = Transition.GENE_ACTIVATE
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    with pytest.raises(ValueError):
        update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)

def test_update_state_fails_given_a_string_as_transition():
    """Verify that the update_state function raises a 'type' error
    given a string as transition.
    """
    active_gene_event = 'Transition.GENE_INACTIVATE'
    inactive_gene_event = Transition.GENE_ACTIVATE
    gene_state = 'active'
    RNA_Protein_state = np.array([0,0])
    with pytest.raises(TypeError):
        update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)



