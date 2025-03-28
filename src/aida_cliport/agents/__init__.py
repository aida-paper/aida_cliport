# credit: https://github.com/cliport/cliport

from aida_cliport.agents.transporter_lang_goal import (
    TwoStreamClipLingUNetLatTransporterAgent,
    DropoutTwoStreamClipLingUNetLatTransporterAgent,
)


names = {
    ###############
    ### cliport ###
    "cliport": TwoStreamClipLingUNetLatTransporterAgent,
    "dropout_cliport": DropoutTwoStreamClipLingUNetLatTransporterAgent,
}
