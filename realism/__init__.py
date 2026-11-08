"""Closing the sim-to-real gap: measure where the simulator's traffic differs from real Argoverse 2 logs, split the difference into
the sensor (tracker noise), the scene mix and the drivers, and patch each with a mechanism fitted to the data.

    stats          one extractor for real and simulated scenes: distributions, tail rates, contexts; the scoreboard
    measure        the tracker-noise model of real logs (fitted from the logs), a model-based smoother for real scenes and the same
                   noise pushed through simulated scenes (measurement parity)
    mix            scene-mix importance weights (a classifier density ratio)
    following      IDM fits on real car-following episodes -> the trait bank the simulator samples drivers from
    discriminator  a classifier two-sample test: can a network tell real from simulated vehicles, and by what?
    refit          the cross-entropy fit of the new knobs on the scoreboard -> the AV2 preset
    replay         log replay: does the patched simulator predict what logged vehicles did next better?
    pipeline       the cached DAG of all of it; api: run / report / preset
"""
