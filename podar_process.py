from math import *
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import nearest_points
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Veh_obj:
    # 基本信息
    type: str = 'car'
    name: str = 'None'
    id: int = 0
    x0: float = 0.
    y0: float = 0.
    phi0: float = 0.  # rad
    a0: float = 0.
    phi_a0: float = 0.  # yaw rate
    speed: float = 0.
    mass: float = -1.
    sensitive: float = -1.
    length: float = -1.
    width: float = -1.
    max_dece: float = 7.5
    # trajectory prediction infos
    pred_steps: int = 31
    x_pred: np.ndarray = np.ndarray(0, )
    y_pred: np.ndarray = np.ndarray(0, )
    v_pred: np.ndarray = np.ndarray(0, )
    phi_pred: np.ndarray = np.ndarray(0, )
    future_position: List[Polygon] = field(default_factory=list)
    virtual_center_x: np.ndarray = np.ndarray(0, )
    virtual_center_y: np.ndarray = np.ndarray(0, )
    # risk evaluation infos at each prediction step
    dis_t: np.ndarray = np.ndarray(0, )  # relative distances 
    time_t: np.ndarray = np.ndarray(0, )  # timestamp
    delta_v: np.ndarray = np.ndarray(0, )  # relative velocity
    abs_v: np.ndarray = np.ndarray(0, )  # speed magnitude
    damage: np.ndarray = np.ndarray(0, )  # damages

    def __post_init__(self):
        """default configurations"""
        assert self.type in ['car', 'tru', 'bic', 'ped'], "vehicle type should be one of {'car', 'tru', 'bic', 'ped'}"
        if self.type == 'car':
            if self.mass == -1.: self.mass = 1.8
            if self.sensitive == -1.: self.sensitive = 1
            if self.length == -1.: self.length = 4.5
            if self.width == -1.: self.width = 1.8
        elif self.type == 'tru':
            if self.mass == -1.: self.mass = 4.5
            if self.sensitive == -1.: self.sensitive = 1
            if self.length == -1.: self.length = 6
            if self.width == -1.: self.width = 1.9
        elif self.type == 'bic':
            if self.mass == -1.: self.mass = 0.09
            if self.sensitive == -1.: self.sensitive = 50
            if self.length == -1.: self.length = 1.65
            if self.width == -1.: self.width = 0.7
        elif self.type == 'ped':
            if self.mass == -1.: self.mass = 0.07
            if self.sensitive == -1.: self.sensitive = 50
            if self.length == -1.: self.length = 0.6
            if self.width == -1.: self.width = 0.6

    def update(self, **kwargs):
        for key, val in kwargs.items():
            assert key in vars(self), '{} is not a class attr'.format(key)
            exec("self.{0}=val".format(key), {'self': self, 'val': val})

@dataclass
class Vehicles:
    ego: Veh_obj = field(default_factory=Veh_obj)
    obj: List[Veh_obj] = field(default_factory=list)

    def set_ego(self, type, **kwargs):
        _o = Veh_obj(type=type)
        for key, val in kwargs.items():
            assert key in vars(_o), '{} is not a class attr'.format(key)
            exec("_o.{0}={1}".format(key, val))
        self.ego = _o

    def add_obj(self, type, **kwargs):
        _o = Veh_obj(type=type, id=len(self.obj))
        for key, val in kwargs.items():
            assert key in vars(_o), '{0} is not a class attr'.format(key)
            exec("_o.{0}={1}".format(key, val))
        self.obj.append(_o)
        return len(self.obj) - 1

def traj_predition(veh: Veh_obj, step_interval=0.1, pred_hori=3):
    '''
    step_interval = 0.1  # prediction step interval
    '''
    x, y, v, phi, a, a_v, L = veh.x0, veh.y0, veh.speed, veh.phi0, veh.a0, veh.phi_a0, veh.length
    x_pre, y_pre, v_pre, phi_pre = [x], [y], [v], [phi]
    veh.pred_steps = int(pred_hori / step_interval) + 1
    
    flag = 0
    for i in range(int(pred_hori / step_interval)):
        v_pre.append(np.clip(v_pre[i] + step_interval * a, 0, None))
        x_pre.append(x_pre[i] + step_interval * v_pre[i] * np.cos(phi_pre[i]))
        y_pre.append(y_pre[i] + step_interval * v_pre[i] * np.sin(phi_pre[i]))
        phi_pre.append(phi_pre[i] + step_interval * v_pre[i] * np.tan(a_v) / L)

    veh.update(x_pred=np.array(x_pre), y_pred=np.array(y_pre), v_pred=np.array(v_pre), phi_pred=np.array(phi_pre))

def get_future_position_shapely(veh: Veh_obj, ego_flag=False):
    traj_x_true, traj_y_true, traj_heading_true, veh_w, veh_l = \
        veh.x_pred, veh.y_pred, veh.phi_pred, veh.width, veh.length
    assert len(traj_x_true) > 0, 'there is no predicted traj'
    shapely_results = []
    beta = np.arctan2(veh_w / 2, veh_l / 2)  # vehicle center-four point angle
    r = np.sqrt(np.power(veh_w, 2) + np.power(veh_l, 2)) / 2  # rotation radius

    x_c1 = traj_x_true + r * np.cos(beta + traj_heading_true)  # top-left
    y_c1 = traj_y_true + r * np.sin(beta + traj_heading_true)
    x_c2 = traj_x_true + r * np.cos(beta - traj_heading_true)  # top-right
    y_c2 = traj_y_true - r * np.sin(beta - traj_heading_true)
    x_c5 = traj_x_true - r * np.cos(beta - traj_heading_true)  # bottom-left
    y_c5 = traj_y_true + r * np.sin(beta - traj_heading_true)
    x_c6 = traj_x_true - r * np.cos(beta + traj_heading_true)  # bottom-right
    y_c6 = traj_y_true - r * np.sin(beta + traj_heading_true)

    for i in range(len(traj_x_true)):
        shapely_results.append(Polygon(((x_c1[i], y_c1[i]),
                                        (x_c2[i], y_c2[i]),
                                        (x_c6[i], y_c6[i]),
                                        (x_c5[i], y_c5[i]))))
    
    if ego_flag:
        virtual_center_x = [traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * 1,
                            traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * -1]
        virtual_center_y = [traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * 1,
                            traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * -1]
    else:            
        virtual_center_x = traj_x_true + veh_l / 2 * np.cos(traj_heading_true) * -1
        virtual_center_y = traj_y_true + veh_l / 2 * np.sin(traj_heading_true) * -1

    veh.update(future_position=shapely_results, virtual_center_x=virtual_center_x, virtual_center_y=virtual_center_y)

def get_risk_to_obj(ego: Veh_obj, obj: Veh_obj, step_interval: float = 0.1, pred_hori=3):
    t_step = int(pred_hori / step_interval)
    dis_t = []
    assert len(obj.future_position) > 0, 'Should get future position first'
    for i in range(t_step + 1):
        dis_t.append(ego.future_position[i].distance(obj.future_position[i]))
    dis_t = np.array(dis_t)

    vx0, vx1 = ego.v_pred * np.cos(ego.phi_pred), obj.v_pred * np.cos(obj.phi_pred)
    vy0, vy1 = ego.v_pred * np.sin(ego.phi_pred), obj.v_pred * np.sin(obj.phi_pred)
    vec_v_x = vx1 - vx0
    vec_v_y = vy1 - vy0
    # ego vehicle use the front and rear points and other vehicle use the rear point
    vec_dir_x_f = ego.virtual_center_x[0] - obj.virtual_center_x
    vec_dir_y_f = ego.virtual_center_y[0] - obj.virtual_center_y
    vec_dir_x_r = ego.virtual_center_x[1] - obj.virtual_center_x
    vec_dir_y_r = ego.virtual_center_y[1] - obj.virtual_center_y
    modd_f = np.linalg.norm([vec_dir_x_f, vec_dir_y_f], axis=0) + 0.00001
    modd_r = np.linalg.norm([vec_dir_x_r, vec_dir_y_r], axis=0) + 0.00001
    vec_dir_x_f, vec_dir_y_f = vec_dir_x_f / modd_f, vec_dir_y_f / modd_f
    vec_dir_x_r, vec_dir_y_r = vec_dir_x_r / modd_r, vec_dir_y_r / modd_r
    
    delta_v_f = vec_v_x * vec_dir_x_f + vec_v_y * vec_dir_y_f
    delta_v_r = vec_v_x * vec_dir_x_r + vec_v_y * vec_dir_y_r
    
    delta_v = np.max([delta_v_f, delta_v_r], axis=0)    

    abs_v = ego.v_pred + obj.v_pred

    time_t = np.linspace(0, pred_hori, t_step + 1)
    dis_t[dis_t < 0] = 0

    obj.update(delta_v=delta_v, abs_v=abs_v, dis_t=dis_t, time_t=time_t)

def step(vehs: Vehicles, step_interval=0.1, pred_hori=3):

    if len(vehs.obj) == 0: return (0, 0, 0)
    traj_predition(vehs.ego, step_interval=step_interval, pred_hori=pred_hori)
    get_future_position_shapely(vehs.ego, ego_flag=True)
    for obj in vehs.obj:
        traj_predition(obj, step_interval=step_interval, pred_hori=pred_hori)
        get_future_position_shapely(obj)
    get_risk_to_obj(vehs.ego, obj, step_interval=step_interval, pred_hori=pred_hori)
