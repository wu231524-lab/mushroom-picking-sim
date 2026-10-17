"""DAY13-verified fixed joint routes for one autonomous mushroom cycle.

The tuples below are copied from the runners that completed the DAY13 Gazebo
flow.  Reverse routes are derived from the same tuples to prevent forward and
return data from drifting apart.  XY alignment remains perception-driven and
therefore intentionally has no fixed route here.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Tuple

from .pick_state import PickState


JointVector = Tuple[float, float, float, float, float, float, float]

JOINT_NAMES = tuple(f"fer_joint{index}" for index in range(1, 8))

HOME_JOINTS: JointVector = (
    -1.8337686924496666e-09,
    -0.7850000007903287,
    3.317435215421938e-10,
    -2.356000000385448,
    -7.098286403106613e-10,
    1.5709999985444636,
    0.7850000049070589,
)

PREGRASP_JOINTS: JointVector = (
    -0.0031409999999993943,
    -0.02973699999999979,
    0.01885400000000015,
    -1.530849,
    0.0005639999999997869,
    1.5012780000000001,
    0.8011420000000005,
)

POST_XY_JOINTS: JointVector = (
    -0.009494263680440262,
    -0.05297959347830694,
    0.011901775034379725,
    -1.5582607593009747,
    0.000665236440065442,
    1.5054232381708905,
    0.787837370726419,
)

Z_APPROACH_WAYPOINTS = (
    (-0.009532676971144472, -0.0697109883347419,
     0.011834223928858842, -1.6303142246175943,
     0.0008579219527679727, 1.5607464143330643,
     0.7876845545494559),
    (-0.00954725153765147, -0.08244456346655787,
     0.011794002603929668, -1.6968820519141976,
     0.001005791331563872, 1.614581480777911,
     0.7875658168703399),
    (-0.009545350262900654, -0.09151277711857424,
     0.011777050019416525, -1.7586685524594683,
     0.0011150131134796653, 1.6673003323406204,
     0.7874781495288253),
    (-0.009532197320527647, -0.09714272424950755,
     0.011779913776028354, -1.816182738438941,
     0.0011891776465236497, 1.7191849146093252,
     0.7874195401582865),
    (-0.009511638373295635, -0.09948950249445178,
     0.011799558775546549, -1.8697980571919541,
     0.00123000040369692, 1.7704535952557947,
     0.7873888010499188),
    (-0.009486563068017012, -0.0986577538218029,
     0.011833185325697194, -1.9197900493925038,
     0.0012377291532115404, 1.8212772833902982,
     0.7873854635112084),
    (-0.009459165672621997, -0.09471644007067095,
     0.011878086858244393, -1.9663612057644329,
     0.0012113806530828243, 1.8717895115932184,
     0.7874097164451882),
    (-0.00944230817886307, -0.09087865190882435,
     0.011909288746799987, -1.9927219259237003,
     0.0011783951076034561, 1.901987783361098,
     0.7874378229643287),
)

LIFT_WAYPOINTS = tuple(reversed(
    (POST_XY_JOINTS,) + Z_APPROACH_WAYPOINTS[:-1]
))

TRANSPORT_WAYPOINTS = (
    (-0.02191235665413507, -0.09466966650154887,
     -0.0024701202138938164, -1.6064559415091024,
     -0.00019657351475296707, 1.5119239279383432,
     0.761019998248908),
    (-0.03399086316894172, -0.13424614647441754,
     -0.01755492723980269, -1.6508000740638296,
     -0.0023119109273383328, 1.5167108177494322,
     0.7338898402078392),
    (-0.045835775176555786, -0.17184672756334923,
     -0.03330853392529122, -1.6916160448562774,
     -0.00565744566831437, 1.5199987271428703,
     0.7064609282802649),
    (-0.05752383173581446, -0.2075671460732378,
     -0.049707367913788894, -1.7291538275009046,
     -0.01020344713551327, 1.5219721774818253,
     0.6787377083543596),
    (-0.06912235052448647, -0.2414708380950803,
     -0.06672973253945838, -1.7636092901741096,
     -0.015912259637719584, 1.5227938401401122,
     0.6507157329714746),
    (-0.08069120622013885, -0.2735956203406663,
     -0.08435419259353698, -1.795137400909512,
     -0.022737682547841757, 1.522610451903083,
     0.6223826887157097),
    (-0.09227407553860405, -0.3039587333435117,
     -0.1025666419286252, -1.8238617111041646,
     -0.030626887414988722, 1.5215569608228996,
     0.5937214721077476),
    (-0.10390198491720243, -0.33256067629337005,
     -0.12135675046392815, -1.8498814051185906,
     -0.0395201890996928, 1.5197594952793179,
     0.5647113791902453),
    (-0.115588703191224, -0.35938857209152997,
     -0.140721172537293, -1.8732766180069584,
     -0.04935258904939558, 1.5173373643613253,
     0.535330811804974),
    (-0.1273371660669952, -0.38441871164865166,
     -0.1606580886579513, -1.8941125479587777,
     -0.06005246842723438, 1.5144044608274365,
     0.505557289150099),
    (-0.13913954457572472, -0.40761886907091704,
     -0.18116719938608472, -1.91244266265148,
     -0.07154159079824975, 1.5110701053481956,
     0.4753684566478119),
    (-0.15097620528911299, -0.4289504770510147,
     -0.2022506779649182, -1.9283112093997945,
     -0.08373548780109392, 1.507439415966337,
     0.44474331816830825),
    (-0.16281641134026884, -0.4483705434107567,
     -0.22391295713696335, -1.9417551897799123,
     -0.09654338580035429, 1.5036133373132927,
     0.4136628622316234),
    (-0.17461911636898125, -0.46583336358819283,
     -0.24616069416559566, -1.9528059012133188,
     -0.10986808987301103, 1.4996883931638507,
     0.3821104327147514),
)

PLACE_DESCENT_WAYPOINTS = (
    (-0.17693432994974728, -0.48087731111188825,
     -0.24782926182413076, -2.0147136082256103,
     -0.11365851656695826, 1.546719091107725,
     0.38502650237773767),
    (-0.17922135051698676, -0.4931573946106227,
     -0.25010362899108185, -2.0737275304838168,
     -0.11738618904730919, 1.5935916378866208,
     0.38752912255885275),
    (-0.1814578125007713, -0.502693938077248,
     -0.2529419655214621, -2.1300508672750404,
     -0.12107295165158523, 1.6404921404115707,
     0.3896612399830584),
    (-0.18362209650946823, -0.5094696337113312,
     -0.2563118138256391, -2.1838346136864906,
     -0.12472800140210674, 1.687590672197011,
     0.3914539229042698),
    (-0.1856912643276964, -0.5134350105321023,
     -0.26018749791010637, -2.2351869542760947,
     -0.128348678083518, 1.735045523885344,
     0.39292810561604213),
    (-0.187639392245673, -0.514512550107388,
     -0.2645480708529307, -2.2841799492883244,
     -0.13192012491647695, 1.783005974388974,
     0.3940954652407239),
    (-0.18943610289279456, -0.5126000555525287,
     -0.26937559631565794, -2.3308543813700338,
     -0.1354139320000118, 1.831613876125621,
     0.39495854534425157),
    (-0.19104518125373515, -0.5075737381372265,
     -0.2746536114802388, -2.375223357996182,
     -0.13878573664703425, 1.8810041977578345,
     0.39551013521954853),
)

VERTICAL_RETREAT_WAYPOINTS = tuple(reversed(
    (TRANSPORT_WAYPOINTS[-1],) + PLACE_DESCENT_WAYPOINTS[:-1]
))

HORIZONTAL_RETURN_WAYPOINTS = tuple(reversed(
    (POST_XY_JOINTS,) + TRANSPORT_WAYPOINTS[:-1]
))


@dataclass(frozen=True)
class FixedTrajectoryRoute:
    state: PickState
    start_joints: JointVector
    waypoints: Tuple[JointVector, ...]
    duration_sec: float
    day13_source: str


FIXED_ROUTES: Mapping[PickState, FixedTrajectoryRoute] = {
    PickState.MOVE_PREGRASP: FixedTrajectoryRoute(
        PickState.MOVE_PREGRASP,
        HOME_JOINTS,
        (PREGRASP_JOINTS,),
        8.0,
        "DAY13 HOME to PREGRASP",
    ),
    PickState.Z_APPROACH: FixedTrajectoryRoute(
        PickState.Z_APPROACH,
        POST_XY_JOINTS,
        Z_APPROACH_WAYPOINTS,
        5.0,
        "day13_vacuum_z_approach_once.py",
    ),
    PickState.LIFT: FixedTrajectoryRoute(
        PickState.LIFT,
        Z_APPROACH_WAYPOINTS[-1],
        LIFT_WAYPOINTS,
        5.0,
        "day13_vacuum_lift_once.py",
    ),
    PickState.TRANSPORT: FixedTrajectoryRoute(
        PickState.TRANSPORT,
        POST_XY_JOINTS,
        TRANSPORT_WAYPOINTS,
        5.0,
        "day13_vacuum_transport_once.py",
    ),
    PickState.PLACE_DESCENT: FixedTrajectoryRoute(
        PickState.PLACE_DESCENT,
        TRANSPORT_WAYPOINTS[-1],
        PLACE_DESCENT_WAYPOINTS,
        5.0,
        "day13_vacuum_place_descent_once.py",
    ),
    PickState.VERTICAL_RETREAT: FixedTrajectoryRoute(
        PickState.VERTICAL_RETREAT,
        PLACE_DESCENT_WAYPOINTS[-1],
        VERTICAL_RETREAT_WAYPOINTS,
        5.0,
        "day13_post_place_vertical_retreat_once.py",
    ),
    PickState.HORIZONTAL_RETURN: FixedTrajectoryRoute(
        PickState.HORIZONTAL_RETURN,
        TRANSPORT_WAYPOINTS[-1],
        HORIZONTAL_RETURN_WAYPOINTS,
        5.0,
        "day13_post_place_horizontal_return_once.py",
    ),
    PickState.RECOVER_PREGRASP: FixedTrajectoryRoute(
        PickState.RECOVER_PREGRASP,
        POST_XY_JOINTS,
        (PREGRASP_JOINTS,),
        5.0,
        "day13_post_xy_to_pregrasp_once.py",
    ),
    PickState.RETURN_HOME: FixedTrajectoryRoute(
        PickState.RETURN_HOME,
        PREGRASP_JOINTS,
        (HOME_JOINTS,),
        8.0,
        "day13_pregrasp_to_home_once.py",
    ),
}


def maximum_joint_step(route: FixedTrajectoryRoute) -> float:
    """Return the largest single-joint change across a fixed route."""
    previous = route.start_joints
    maximum = 0.0
    for waypoint in route.waypoints:
        maximum = max(
            maximum,
            max(abs(after - before) for before, after in zip(
                previous, waypoint
            )),
        )
        previous = waypoint
    return maximum


def validate_fixed_routes() -> dict:
    """Validate dimensions, finiteness, and exact route continuity."""
    for state, route in FIXED_ROUTES.items():
        if state != route.state:
            raise ValueError("fixed_route_state_mismatch")
        if len(route.start_joints) != len(JOINT_NAMES):
            raise ValueError("fixed_route_start_dimension_invalid")
        if not route.waypoints:
            raise ValueError("fixed_route_empty")
        for waypoint in route.waypoints:
            if len(waypoint) != len(JOINT_NAMES):
                raise ValueError("fixed_route_waypoint_dimension_invalid")
            if not all(math.isfinite(value) for value in waypoint):
                raise ValueError("fixed_route_waypoint_not_finite")
    return {
        "valid": True,
        "fixed_route_count": len(FIXED_ROUTES),
        "fixed_trajectory_count": sum(
            len(route.waypoints) for route in FIXED_ROUTES.values()
        ),
        "dynamic_xy_trajectory_limit": 5,
        "maximum_total_trajectory_count": (
            sum(len(route.waypoints) for route in FIXED_ROUTES.values())
            + 5
        ),
    }
