# Finding intersections in any map

`xnet/junctions.py`. Takes an OpenStreetMap-style road graph (nodes, ways, tags) and returns the *logical* intersections: one per place where
drivers experience a junction, with its legs, its kind and its control tags, in O(n log n). `av_world.make("map.osm")`, `av-world import-osm`
and `xnet.osm.import_osm` run it first, so the road network builder sees a cleaned map.

## Why the obvious rule is wrong

"A node where three or more road pieces meet is an intersection" is right for the textbook grid and wrong in five common ways:

| situation in the map data | what the degree-3 rule says | what it is |
|---|---|---|
| a divided road crossing a street: two carriageways 10-20 m apart | two T-junctions | one intersection |
| a pair of T-junctions a few metres apart (an offset crossing) | two | one |
| a roundabout: a ring of nodes, one per entry | one per entry | one junction |
| a bridge sharing a node with the road below (`layer`, `bridge`, `tunnel`) | a junction | no junction |
| a pedestrian signal or crossing tagged on a node mid-block | (degree 2: none) but the *tag* then mislabels the next junction | not a junction; the signal tag belongs to real junctions only |

## The algorithm

```
detect(nodes, ways, node_tags):
    xy = project(nodes) to local metres

    # 1. roundabouts: contract each ring to one node
    for each way w with junction=roundabout, or a closed one-way ring of >= 4 nodes:
        centre = mean(xy of ring nodes);  give it a new id
        map every ring node to that id
    drop the ring ways
    rewrite every other way's node list through the map; collapse repeats

    # 2. grade separation: ways at different levels never meet at a shared node
    level(w) = layer(w) + (1 if bridge) - (1 if tunnel)
    for each node n shared by ways of more than one level:
        in each way replace n by the copy (n, level of that way)

    # 3. the road graph
    adj[n] = [(neighbour, way)] for every consecutive pair of nodes in every way
    deg[n] = len(adj[n])
    C = { n : deg[n] >= 3 }

    # 4. cluster candidates into logical intersections (union-find)
    for each pair of candidates closer than NEAR (12 m) in a straight line      # KD-tree range query
        union them
    for each candidate n, for each road piece leaving it:
        walk along the road through degree-2 nodes until a non-degree-2 node or JOIN (30 m)
        if that node is a candidate and the road between is shorter than JOIN:
            union them                                                           # divided roads, offset T's

    # 5. legs and kind of every cluster
    for each cluster with member nodes M, centre c = mean(xy[M]):
        for each road piece leaving M:
            walk LEG_M (18 m) along the road; bearing = direction from c to that point
        merge legs less than 30 deg apart that have the same street name, no name, or are both one-way
            # the two carriageways of one street are one road
        drop the cluster if fewer than 3 merged legs (and no roundabout)
        kind = roundabout | T (3 legs) | cross (4) | multi (5+)

    # 6. tags
    for each node with highway=traffic_signals or highway=stop:
        j = nearest junction centre within 30 m (KD-tree)
        if exists: j.signal / j.stop = true         # the builder then signalises exactly these
        else: drop the tag                          # a mid-block pedestrian signal
    return junctions, cleaned nodes and ways, tag positions
```

Every step is a pass over the nodes, a KD-tree query, or a bounded walk (at most 30 m) from a candidate, so the cost is O(n log n). The
union-find has path compression; the walks only start at degree-3 nodes.

## What the builder does with it

`xnet.network.build(hints=...)` receives the positions of tagged signals and stop signs. A junction within 30 m of a signal tag is
signalised; when an extract has three or more signal tags the builder stops guessing signals from road class and trusts the tags (an extract
without tags falls back to the old rule: both roads at least secondary and tertiary). The roundabout becomes one junction whose legs are
equal-ranked, which the simulator treats as an all-way stop (a yield approximation).

## How good is it

Three checks, with different strength.

1. **A made-up city with a known answer** (`xnet/synth_city.py`, `tests/test_junctions.py`): a grid with divided arterials, offset T's,
   roundabouts, overpasses with and without a shared node, mid-block signal and crossing tags. Six cities of 100 crossings each:
   precision 1.00 and recall 1.00, against 0.55-0.67 precision (F1 0.75) for the degree-3 rule. This is the weakest evidence: the awkward
   cases were chosen by me, alongside the algorithm. It shows that each case is handled, not that the real world has no others.
2. **Real Oakland roads (Overture), against the earlier miner** (`pgh/mine.py`, a different algorithm on the same data): 186 junctions found
   inside the area against its 165; F1 0.92 (precision 0.87, recall 0.98) at a 25 m match radius, against F1 0.89 for the degree-3 rule.
   This is agreement between two algorithms, not accuracy. The 24 detections the miner does not have are crossings of named streets (for
   example Wandless Street with Webster Avenue); some are probably real junctions that the miner dropped for lying close to a neighbour,
   some may be duplicates. I did not check them against a ground-truth survey.
3. **Real signal tags: not tested.** The container could not reach the OpenStreetMap servers, and Overture carries no signal tags, so
   the signal logic (junction control from tags) is verified only on the synthetic city.

Speed, single core, detection only (parsing excluded):

| map | nodes | time |
|---|---|---|
| Oakland (421 segments) | 3,252 | 0.02 s |
| synthetic, 100 x 100 crossings | 26,905 | 0.41 s |
| synthetic, 200 x 200 crossings | 110,270 | 2.3 s |

About 50,000 nodes per second, linear in size. A city of a few hundred thousand nodes takes seconds.

## Limits

* Turn restrictions, lane counts, speed limits other than `maxspeed`, and signal timings are not read; lanes come from road class.
* A roundabout becomes an all-way stop, which is slower than a real roundabout and removes the circulating conflict.
* Interchanges with ramps (`*_link` roads) are dropped by the builder; grade-separated crossings are kept apart but their ramps are not modelled.
* A clustering radius of 30 m is a choice: a very wide median (more than 30 m) leaves a divided road as two junctions.
* For lane-level maps (Argoverse 2, Waymo) the junctions come from the `is_intersection` flag (Argoverse) or lane crossings (Waymo);
  see DATA_IMPORT.md.
