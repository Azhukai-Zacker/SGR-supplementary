# -*- coding: utf-8 -*-
"""Run the hindsight entropy top-K SGR diagnostic on InternVL COCO.

This is not an online deployable trigger because it first records a baseline
entropy trace and then reruns decoding at the highest-entropy positions.
"""

from internvl_entropy_trigger_coco import main


if __name__ == "__main__":
    main(default_mode="oracle")
