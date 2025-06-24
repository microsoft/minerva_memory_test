from task.search import *
from task.recall_and_edit import *
from task.match_and_compare import *
from task.spot_the_differences import *
from task.compute_on_sets_and_lists import *
from task.stateful_processing import *
from task.composite import *


TASK_CLASSES = {
    "search": [
        StringSearchWord,
        # StringSearchGibberish, # Uncomment if you want to skip task
        StringSearchSequence,
        KeyValueSearch,
        BatchKeyValueSearch
    ],
    "recall_and_edit": [
        {"class": Snapshot, "params": {"context_type": "unique_words"}},
        {"class": Snapshot, "params": {"context_type": "random_numbers"}}, 
        ReplaceAll,
        ReplaceAllXToNull,
        OverwritePositions,
        OverwritePositionsNthToNull,
        FunctionalUpdates
    ],
    "match_and_compare": [
        ComparePositions,
        FindDuplicates,
        Count,
        CheckAssociation
    ],
    "spot_the_differences": [
        CompareTwoLists,
        IdentifyOddGroup,
        PatchDifference
    ],
    "compute_on_sets_and_lists": [
        GroupMembership,
        GroupAssociation,
        AlternatingGroupAssociation,
        Iterate
    ],
    "stateful_processing": [
        QuantityState,
        SetState
    ],
    "composite": [
        ProcessingDataBlocks,
        TheoryOfMind
    ]
}