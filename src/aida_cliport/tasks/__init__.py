# credit: https://github.com/cliport/cliport

"""Ravens tasks."""

from aida_cliport.tasks.packing_shapes import PackingShapesOriginal
from aida_cliport.tasks.packing_shapes import PackingSeenShapes
from aida_cliport.tasks.packing_shapes import PackingUnseenShapes
from aida_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsOriginalSeq
from aida_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsOriginalSeq
from aida_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsOriginalGroup
from aida_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsOriginalGroup
from aida_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from aida_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from aida_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsGroup
from aida_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsGroup
from aida_cliport.tasks.put_block_in_bowl import PutBlockInBowlSeenColors
from aida_cliport.tasks.put_block_in_bowl import PutBlockInBowlUnseenColors
from aida_cliport.tasks.put_block_in_bowl import PutBlockInBowlFull

names = {
    # goal conditioned
    "packing-shapes-original": PackingShapesOriginal,
    "packing-shapes": PackingSeenShapes,
    "packing-seen-shapes": PackingSeenShapes,
    "packing-unseen-shapes": PackingUnseenShapes,
    "packing-seen-google-objects-original-seq": PackingSeenGoogleObjectsOriginalSeq,
    "packing-unseen-google-objects-original-seq": PackingUnseenGoogleObjectsOriginalSeq,
    "packing-seen-google-objects-original-group": PackingSeenGoogleObjectsOriginalGroup,
    "packing-unseen-google-objects-original-group": PackingUnseenGoogleObjectsOriginalGroup,
    "packing-seen-google-objects-seq": PackingSeenGoogleObjectsSeq,
    "packing-unseen-google-objects-seq": PackingUnseenGoogleObjectsSeq,
    "packing-seen-google-objects-group": PackingSeenGoogleObjectsGroup,
    "packing-unseen-google-objects-group": PackingUnseenGoogleObjectsGroup,
    "put-block-in-bowl-seen-colors": PutBlockInBowlSeenColors,
    "put-block-in-bowl-unseen-colors": PutBlockInBowlUnseenColors,
    "put-block-in-bowl-full": PutBlockInBowlFull,
}
