import numpy as np
import numpy.linalg as la

nparticles = 2000
x = -1 + 2*np.random.rand(2, nparticles)
x = np.sign(x)*np.abs(x)**1.9
x = (1.4 + x) % 2 - 1

# {{{ tree node

def do_boxes_intersect(bl, tr):
    (bl1, tr1) = bl
    (bl2, tr2) = tr
    (dimension,) = bl1.shape
    for i in range(0, dimension):
        if max(bl1[i], bl2[i]) > min(tr1[i], tr2[i]):
            return False
    return True


def make_children(bottom_left, top_right, allchildren):
    import numpy

    (dimensions,) = bottom_left.shape

    half = (top_right - bottom_left) / 2.

    def do(dimension, pos):
        if dimension == dimensions:
            origin = bottom_left + pos*half
            child = TreeNode(origin, origin + half)
            allchildren.append(child)
            return child
        else:
            pos[dimension] = 0
            first = do(dimension + 1, pos)
            pos[dimension] = 1
            second = do(dimension + 1, pos)
            return [first, second]

    return do(0, numpy.zeros((dimensions,), numpy.float64))


class TreeNode:
    """This class represents one node in a spatial binary tree.
    It automatically decides whether it needs to create more subdivisions
    beneath itself or not.

    :ivar elements: a list of tuples *(element, bbox)* where bbox is again
      a tuple *(lower_left, upper_right)* of :class:`numpy.ndarray` instances
      satisfying *(lower_right <= upper_right).all()*.
    """

    def __init__(self, bottom_left, top_right, max_el_per_box=10):
        """:param bottom_left: A :mod: 'numpy' array of the minimal coordinates
        of the box being partitioned.
        :param top_right: A :mod: 'numpy' array of the maximal coordinates of
        the box being partitioned."""

        self.elements = []

        self.bottom_left = bottom_left
        self.top_right = top_right
        self.center = (bottom_left + top_right) / 2

        # As long as children is None, there are no subdivisions
        self.children = None
        self.elements = []

        self.max_el_per_box = max_el_per_box

    def insert(self, element, bbox):
        """Insert an element into the spatial tree.

        :param element: the element to be stored in the retrieval data
        structure.  It is treated as opaque and no assumptions are made on it.

        :param bbox: A bounding box supplied as a tuple *lower_left,
        upper_right* of :mod:`numpy` vectors, such that *(lower_right <=
        upper_right).all()*.

        Despite these names, the bounding box (and this entire data structure)
        may be of any dimension.
        """

        def insert_into_subdivision(element, bbox):
            for child in self.all_children:
                if do_boxes_intersect(
                        (child.bottom_left, child.top_right), bbox):
                    child.insert(element, bbox)

        (dimensions,) = self.bottom_left.shape
        if self.children is None:
            # No subdivisions yet.
            if len(self.elements) > self.max_el_per_box:
                # Too many elements. Need to subdivide.
                self.all_children = []
                self.children = make_children(
                        self.bottom_left, self.top_right,
                        self.all_children)

                # Move all elements from the full child into the new finer ones
                for el, el_bbox in self.elements:
                    insert_into_subdivision(el, el_bbox)

                # Free up some memory. Elements are now stored in the
                # subdivision, so we don't need them here any more.
                del self.elements

                insert_into_subdivision(element, bbox)
            else:
                # Simple:
                self.elements.append((element, bbox))
        else:
            # Go find which sudivision to place element
            insert_into_subdivision(element, bbox)

    def generate_matches(self, point):
        if self.children:
            # We have subdivisions. Use them.
            (dimensions,) = point.shape
            child = self.children
            for dim in range(dimensions):
                if point[dim] < self.center[dim]:
                    child = child[0]
                else:
                    child = child[1]

            for result in child.generate_matches(point):
                yield result
        else:
            # We don't. Perform linear search.
            for el, bbox in self.elements:
                yield el

    def plot(self, **kwargs):
        import matplotlib.pyplot as pt
        import matplotlib.patches as mpatches
        from matplotlib.path import Path

        el = self.bottom_left
        eh = self.top_right
        pathdata = [
            (Path.MOVETO, (el[0], el[1])),
            (Path.LINETO, (eh[0], el[1])),
            (Path.LINETO, (eh[0], eh[1])),
            (Path.LINETO, (el[0], eh[1])),
            (Path.CLOSEPOLY, (el[0], el[1])),
            ]

        codes, verts = zip(*pathdata)
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, **kwargs)
        pt.gca().add_patch(patch)

        if self.children:
            for i in self.all_children:
                i.plot(**kwargs)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id == other.id


# }}}

# {{{ tree utilities

def levels_to_boxes(tree, base_level=0):
    if tree.children is None:
        return {base_level: [tree]}
    else:
        result = {}
        for ch in tree.all_children:
            for lev, nodes in levels_to_boxes(ch, base_level+1).items():
                result[lev] = result.get(lev, []) + nodes
        return result


def list_boxes(tree):
    result = [tree]
    if tree.children is not None:
        for ch in tree.all_children:
            result.extend(list_boxes(ch))

    return result

# }}}

# {{{ data setup

bl = np.min(x, axis=-1)
tr = np.max(x, axis=-1)
tree = TreeNode(bl, tr, max_el_per_box=10)
for i in range(nparticles):
    tree.insert(i, (x[:, i], x[:, i]))

boxes = list_boxes(tree)
for i, box in enumerate(boxes):
    box.id = i

from random import choice
target_boxes = [
        choice(boxes) for boxes in levels_to_boxes(tree).values()]

for tb in target_boxes:
    assert tb.children is None, "target_box is not a leaf"

# }}}

# {{{ interaction collector

class InteractionCollector:
    def __init__(self):
        self.box_to_sources = {}

    def add_sources(self, tgt_box, sources):
        tgt_sources = self.box_to_sources.setdefault(tgt_box.id, set())
        adding = set(sources)
        if tgt_sources & adding:
            raise RuntimeError(
                    "box %d already has contribution from sources %s"
                    % tgt_box.id, ", ".join(
                        str(s) for s in tgt_sources & adding))
        tgt_sources.update(sources)

    def evaluate_multipole(self, tgt_box, src_box):
        assert tgt_box.children is None

        src_rad = (src_box.top_right - src_box.bottom_left)[0] / 2
        tgt_rad = (tgt_box.top_right - tgt_box.bottom_left)[0] / 2
        R = la.norm(tgt_box.center - src_box.center, 2)
        if R < 3*max(src_rad, tgt_rad) - 1e-10:
            raise RuntimeError("boxes not well-separated "
                "(src radius: %g, tgt radius: %g, dist: %g)"
                % (src_rad, tgt_rad, R))

        if src_box.children is not None:
            for ch in src_box.all_children:
                self.evaluate_multipole(tgt_box, ch)
        else:
            self.add_sources(tgt_box, [el for el, bbox in src_box.elements])

    translate_multipole = evaluate_multipole

    def compute_directly(self, tgt_box, src_box):
        self.add_sources(tgt_box, [el for el, bbox in src_box.elements])

interactions = InteractionCollector()

# }}}



#
#
# Code for implementing interaction list for multi-level Barnes-Hut tree code
#
#
import math
import numpy as np

def getDims(box):
    width = np.abs(box.top_right[0] - box.bottom_left[0])
    height= np.abs(box.top_right[1] - box.top_right[1])
    return (width, height)

def maxdist(dx, dy):
    return min(dx,dy)

def relBoxLevel(tbox,box):
    (w1, h1) = getDims(tbox)
    (w2, h2) = getDims(box)
    return round(math.log2(w1/w2))

def minBoxDist(box1,box2):
    (w1,h1) = getDims(box1)
    (w2,h2) = getDims(box2)
    return np.abs(box1.center - box2.center) - 0.5*(w1+w2)

def recursiveInteractionList( target_box, tree_node, interactions ):
    (wt, _) = getDims(target_box)
    if tree_node.children is None: # a leaf node that should be checked
        # check what list node should be in
        reld = relBoxLevel(target_box,tree_node)
        d = minBoxDist(target_box,tree_node)
        if d == 0:
            # need to modify this
            interactions.compute_directly(target_box,tree_node)
        else:
            # need to modify this
            interactions.compute_directly(target_box,tree_node)

    else: # not a leaf node

        # recursively check children of current tree node
        for cbox in tree_node.all_children:
            recursiveInteractionList(target_box, cbox, interactions)

        # check if this box could be in list 2



# loop through all the target boxes
for tbox in target_boxes:

    # iterate through tree to find each leaf
    recursiveInteractionList(tbox, tree, interactions)

