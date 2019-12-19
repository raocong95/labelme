import functools
import numpy as np

PI = 3.1415926543

# caculate horizon view angle given pixel x coord
def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi

# caculate vertical view angle given pixel y coord
def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

# from view angle and z value to x,y in real world
def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y


def pano_connect_points(p1, p2, z=-50, w=1024, h=512, last = False):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if ((abs(p1[0] - p2[0]) < w / 2) and (not last)):
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)


    # coorus = coorx2u(coorxs, w)
    # coorvs = coory2v(coorys, h)
    # rx, ry = uv2xy(coorus, coorvs, z)
    # plt.plot(rx, ry, 'r')
    # plt.show()


    return np.stack([coorxs, coorys], axis=-1)

def calc_vertical_point(p1, p2, p3_x, w=1024, h=512):
    ## p2p1 vertical to p3p1
    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    assert((p1[1]-0.5*h)*(p2[1]-0.5*h)>0)
    z = -50
    if p1[1]-0.5*h>0:
        z = 50


    u3 = coorx2u(p3_x, w)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    l1 = np.sqrt(x1*x1+y1*y1)
    l2 = np.sqrt(x2*x2+y2*y2)
    alpha = u2-u1
    beta = u3-u2
    ## (l1^2+l2^2-2*l1*l2*cos(alpha)) + (l2^2+l3^2-2*l2*l3*cos(beta)) = (l1^2+l3^2-2*l1*l3*cos(alpha+beta))
    l3 = (l2*l2-l1*l2*np.cos(alpha))/(l2*np.cos(beta) - l1*np.cos(alpha+beta))

    v3 = np.arctan(z/l3)
    p3_y = v2coory(v3, h=512)
    return p3_y

def calc_vertical_points(p1, p2, p3_xs, w=1024, h=512):
    ## p1p2 vertical to p3p2
    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    #assert((p1[1]-0.5*h)*(p2[1]-0.5*h)>0)

    z = -50
    if p1[1]-0.5*h>0:
        z = 50

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)
    l1 = np.sqrt(x1*x1+y1*y1)
    l2 = np.sqrt(x2*x2+y2*y2)

    p3_ys = []

    for p3_x in p3_xs:
        u3 = coorx2u(p3_x, w)
        alpha = u2-u1
        beta = u3-u2
        ## (l1^2+l2^2-2*l1*l2*cos(alpha)) + (l2^2+l3^2-2*l2*l3*cos(beta)) = (l1^2+l3^2-2*l1*l3*cos(alpha+beta))
        l3 = (l2*l2-l1*l2*np.cos(alpha))/(l2*np.cos(beta) - l1*np.cos(alpha+beta))
        v3 = np.arctan(z/l3)
        p3_y = v2coory(v3, h=512)
        p3_ys.append(p3_y)
    p3_ys = np.array(p3_ys)
    return p3_ys






# pixel x to u
def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI

# pixel y to v
def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI

#pixel xy to room xy
def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])

def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])





def sort_xy_filter_unique(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    length = (max(xs) + 1).astype(int)
    xs1 = np.arange(0, length)
    ys1 = np.zeros(length)
    ys1[np.round(xs).astype(int)] = ys[np.arange(len(xs))]
    return xs1, ys1



def bon_line(cor, side = True, W=1024, H=512):
    if len(cor) < 2:
        return []

    if len(cor) == 2 and abs(cor[0][0]-cor[1][0])<1.0e-6:
        return []


    z_value = 50
    if cor[0][1] < H / 2:
        z_value = -50
    bon_x, bon_y = [], []
    n_cor = len(cor)
    for i in range(n_cor):
        if i < n_cor-1 or side:
            xys = pano_connect_points(cor[i], cor[(i + 1) % n_cor], z=z_value, w=1024, h=512,
                                                  last=(i == n_cor - 1))
            bon_x.extend(xys[:, 0])
            bon_y.extend(xys[:, 1])

    res_x = np.array(bon_x)
    res_y = np.array(bon_y)
    return np.concatenate((res_x[:, None], res_y[:, None]), axis = 1)



if __name__ == '__main__':
    a = np.array([10, 100])
    b = np.array([20, 120])
    c = calc_vertical_point(a, b, 23, w=1024, h=512)
    d = calc_vertical_point(b, a, 23, w=1024, h=512)
    c = calc_vertical_points(a, b, np.array([23, 25, 28]), w=1024, h=512)






    a = 20



