"""Geometry/HSV/LBP baseline. Calculations use float64; storage uses float32."""

import math

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

EPS = 1e-8
DIMS = (7, 10, 10, 42, 50)
BROW_R = [(70, 46), (63, 53), (105, 52), (66, 65), (107, 55)]
BROW_L = [(336, 285), (296, 295), (334, 282), (293, 283), (300, 276)]
ROI_ORDER = ['forehead', 'cheek_left', 'cheek_right', 'eyes', 'mouth']
NAMES = [
    ['height_width', 'jaw_width', 'upper_width', 'chin_length', 'jaw_upper', 'chin_angle', 'asymmetry'],
    ['eye_distance', 'eye_width_left', 'eye_width_right', 'eye_aspect_left', 'eye_aspect_right',
     'eye_y', 'brow_gap_left', 'brow_gap_right', 'brow_angle_left', 'brow_angle_right'],
    ['nose_length', 'nose_width', 'nose_aspect', 'mouth_width', 'mouth_height', 'upper_lip',
     'lower_lip', 'nose_mouth', 'mouth_chin', 'eye_nose'],
    ['median_h_shift', 'median_s', 'median_v', 'std_h_shift', 'std_s', 'std_v'] +
    [f'hs_{h}_{s}' for h in range(8) for s in range(4)] + [f'v_{i}' for i in range(4)],
    [f'{roi}_lbp_{i}' for roi in ROI_ORDER for i in range(10)],
]
CONFIG = dict(min_pixels=25, cheek_axes=[.075, .065], forehead_center=-.22,
              forehead_axes=[.15, .09], eye_margin=[.03, .04], mouth_margin=[.05, .06],
              part_dilation=2, texture_erosion=1, lab_distance=20, forehead_fraction=.5,
              lbp=dict(P=8, R=1, method='uniform'), hsv_bins=[8, 4, 4],
              mask_rasterization='pixel integer centers; closed contour polygons; 3x3 erosion',
              forehead_color_pixels='accepted Lab pixels only')


class FeatureError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def ids(edges):
    return sorted({i for e in edges for i in (e.start, e.end)})


def frame(points, connections):
    if points.shape != (468, 2) or not np.isfinite(points).all():
        raise FeatureError('invalid_geometry', 'Expected 468 finite 2D landmarks')
    r = points[ids(connections.FACE_LANDMARKS_RIGHT_EYE)].mean(0)
    l = points[ids(connections.FACE_LANDMARKS_LEFT_EYE)].mean(0)
    br = points[np.array(BROW_R)].mean(1)
    bl = points[np.array(BROW_L)].mean(1)
    g = (br.mean(0) + bl.mean(0)) / 2
    delta = l - r
    if np.linalg.norm(delta) <= EPS:
        raise FeatureError('invalid_geometry', 'Coincident eye centers')
    u = delta / np.linalg.norm(delta)
    v = np.array([-u[1], u[0]])
    if (points[152] - g) @ v < 0:
        v *= -1
    basis = np.column_stack((u, v))
    xy = (points - g) @ basis
    wf = abs(xy[454, 0] - xy[234, 0])
    h = xy[152, 1]
    return dict(g=g, basis=basis, xy=xy, wf=wf, h=h, r=r, l=l,
                br=br, bl=bl, eye=(r + l)/2, mouth=(points[61] + points[291])/2)


def geometry(points, connections):
    f = frame(points, connections)
    q, wf, h = f['xy'], f['wf'], f['h']
    def project(p):
        return (p - f['g']) @ f['basis']
    def positive(*values):
        if any(x <= EPS for x in values):
            raise FeatureError('invalid_geometry', 'Nonpositive geometric denominator')
    positive(wf, h)
    if wf <= 20 or h <= 20:
        raise FeatureError('invalid_geometry', 'Face width or height <= 20 pixels')
    wj, wu = abs(q[397, 0]-q[172, 0]), abs(q[356, 0]-q[127, 0])
    er, el = abs(q[133, 0]-q[33, 0]), abs(q[263, 0]-q[362, 0])
    hr, hl = abs(q[145, 1]-q[159, 1]), abs(q[374, 1]-q[386, 1])
    ln, wn = q[2, 1]-q[168, 1], abs(q[327, 0]-q[98, 0])
    wm, hm = abs(q[291, 0]-q[61, 0]), q[17, 1]-q[0, 1]
    positive(wu, er, el, wn)
    my, ey = project(f['mouth'])[1], project(f['eye'])[1]
    lengths = [ln, hm, q[13, 1]-q[0, 1], q[17, 1]-q[14, 1],
               my-q[2, 1], q[152, 1]-my, q[2, 1]-ey]
    if any(x < 0 for x in lengths):
        raise FeatureError('invalid_geometry', 'Negative nose/lip/mouth/chin measurement')
    a, b = points[172]-points[152], points[397]-points[152]
    positive(np.linalg.norm(a), np.linalg.norm(b))
    angle = np.arccos(np.clip(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))
    pairs = [(234,454),(172,397),(127,356),(33,263),(133,362),(61,291)]
    asymmetry = np.mean([np.linalg.norm(q[a] - q[b]*[-1, 1]) for a,b in pairs])/wf
    def brow_angle(brow):
        endpoints = project(brow[[0, -1]])
        endpoints = endpoints[np.argsort(endpoints[:, 0])]
        d = endpoints[1]-endpoints[0]
        return np.arctan2(d[1], d[0])
    f1 = [h/wf, wj/wf, wu/wf, (h-my)/h, wj/wu, angle, asymmetry]
    f2 = [np.linalg.norm(f['l']-f['r'])/wf, el/wf, er/wf, hl/el, hr/er, ey/h,
          (project(f['l'])[1]-project(f['bl'].mean(0))[1])/h,
          (project(f['r'])[1]-project(f['br'].mean(0))[1])/h,
          brow_angle(f['bl']), brow_angle(f['br'])]
    f3 = [ln/h, wn/wf, ln/wn, wm/wf, hm/h, lengths[2]/h, lengths[3]/h,
          lengths[4]/h, lengths[5]/h, lengths[6]/h]
    return [np.array(x, dtype=np.float64) for x in (f1, f2, f3)]


def polygon_mask(points, edges, shape):
    # Official contours may comprise several disjoint loops (e.g. lips).
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge.start, set()).add(edge.end)
        adjacency.setdefault(edge.end, set()).add(edge.start)
    mask = np.zeros(shape, np.uint8)
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        ordered, prev, current = [], None, start
        while current not in ordered:
            ordered.append(current)
            unseen.discard(current)
            choices = sorted(adjacency[current] - ({prev} if prev is not None else set()))
            if not choices:
                break
            prev, current = current, choices[0]
        cv2.fillPoly(mask, [np.rint(points[ordered]).astype(np.int32)], 1)
    return mask.astype(bool)


def rois(points, valid, connections):
    f = frame(points, connections)
    wf, h, q = f['wf'], f['h'], f['xy']
    yy, xx = np.indices(valid.shape)
    coords = (np.stack((xx, yy), axis=-1) - f['g']) @ f['basis']
    face = polygon_mask(points, connections.FACE_LANDMARKS_FACE_OVAL, valid.shape) & valid.astype(bool)
    def ellipse(center, rx, ry):
        c = (center-f['g']) @ f['basis']
        return (((coords[...,0]-c[0])/rx)**2 + ((coords[...,1]-c[1])/ry)**2 <= 1) & face
    def rectangle(selected, dx, dy):
        p = q[selected]
        low, high = p.min(0)-[dx,dy], p.max(0)+[dx,dy]
        return ((coords >= low) & (coords <= high)).all(-1) & face
    parts = ['LEFT_EYE','RIGHT_EYE','LEFT_EYEBROW','RIGHT_EYEBROW','NOSE','LIPS']
    exclude = np.zeros(valid.shape, bool)
    for part in parts:
        exclude |= polygon_mask(points, getattr(connections, 'FACE_LANDMARKS_'+part), valid.shape)
    exclude = cv2.dilate(exclude.astype(np.uint8), np.ones((5,5),np.uint8)).astype(bool)
    masks = dict(
        forehead=ellipse(f['g']-.22*h*f['basis'][:,1], .15*wf, .09*h),
        cheek_left=ellipse((points[454]+points[2])/2, .075*wf, .065*h) & ~exclude,
        cheek_right=ellipse((points[234]+points[2])/2, .075*wf, .065*h) & ~exclude,
        eyes=rectangle(sorted(set().union(*(ids(getattr(connections,'FACE_LANDMARKS_'+part))
                      for part in parts[:4]))), .03*wf, .04*h),
        mouth=rectangle(ids(connections.FACE_LANDMARKS_LIPS), .05*wf, .06*h))
    return masks, f


def check_pixels(mask, name):
    if np.count_nonzero(mask) < CONFIG['min_pixels']:
        raise FeatureError('invalid_roi', f'{name}: fewer than 25 valid pixels')


def color(rgb, masks):
    for name in ['cheek_left', 'cheek_right']:
        check_pixels(masks[name], name)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)/[180,255,255]
    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2LAB).astype(np.float64)
    reference = np.median(lab[masks['cheek_left'] | masks['cheek_right']], axis=0)
    accepted = masks['forehead'] & (np.linalg.norm(lab-reference, axis=-1) <= 20)
    use_forehead = accepted.sum() >= 25 and accepted.sum() >= .5*masks['forehead'].sum()
    def vector(mask):
        pixels = hsv[mask]
        shifted = pixels.copy()
        shifted[:,0] = (shifted[:,0]+.5)%1
        stats = np.r_[np.median(shifted, axis=0), np.std(shifted, axis=0, ddof=0)]
        hs = np.histogram2d(pixels[:,0], pixels[:,1], bins=[8,4], range=[[0,1],[0,1]])[0].ravel()
        value = np.histogram(pixels[:,2], bins=4, range=(0,1))[0]
        return np.r_[stats, hs/hs.sum(), value/value.sum()]
    raw = (vector(masks['cheek_left'])+vector(masks['cheek_right']))*(.4 if use_forehead else .5)
    if use_forehead:
        raw += .2*vector(accepted)
    blocks = [raw[:6], raw[6:38], raw[38:]]
    if any(np.linalg.norm(block) <= EPS for block in blocks):
        raise FeatureError('invalid_roi', 'Zero-norm F4 block')
    matching = np.concatenate([block/np.linalg.norm(block)/math.sqrt(3) for block in blocks])
    return raw, matching, dict(forehead_color_used=bool(use_forehead),
                               forehead_accepted_pixels=int(accepted.sum()))


def extract(rgb, source_points, aligned_points, valid, connections):
    vectors = geometry(source_points, connections)
    masks, f = rois(aligned_points, valid, connections)
    raw4, match4, metadata = color(rgb, masks)
    lbp = local_binary_pattern(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), P=8, R=1, method='uniform')
    histograms, centers = [], {}
    for name in ROI_ORDER:
        mask = cv2.erode(masks[name].astype(np.uint8), np.ones((3,3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        check_pixels(mask, 'F5 '+name)
        centers[name] = mask
        hist = np.bincount(lbp[mask].astype(int), minlength=10).astype(np.float64)
        histograms.append(hist/hist.sum())
    vectors += [raw4, np.concatenate(histograms)]
    for vector, dimension in zip(vectors, DIMS):
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise FeatureError('invalid_geometry', 'Unexpected vector dimension or nonfinite values')
    metadata.update(roi_pixels={k:int(v.sum()) for k,v in masks.items()},
                    lbp_centers={k:int(v.sum()) for k,v in centers.items()})
    return vectors, match4, masks, metadata


def overlay(rgb, points, valid, connections):
    masks, f = rois(points, valid, connections)
    result = rgb.copy()
    palette = [(255,170,0),(0,220,255),(255,80,180),(80,255,80),(180,100,255)]
    for name, color_value in zip(ROI_ORDER, palette):
        mask = masks[name]
        result[mask] = (.55*result[mask]+.45*np.array(color_value)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color_value, 1)
    for index, axis in enumerate(f['basis'].T):
        a = tuple(np.rint(f['g']).astype(int))
        b = tuple(np.rint(f['g']+45*axis).astype(int))
        cv2.arrowedLine(result, a, b, (255,0,0) if index==0 else (0,100,255), 1)
    return result
