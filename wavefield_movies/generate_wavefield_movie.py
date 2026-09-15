#!/usr/bin/env python3
import os

# Prevent excessive threading on the login node. Set before scientific imports.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import math, os, re, subprocess, yaml, imageio_ffmpeg, matplotlib
from collections import defaultdict
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, default_fillvals
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay

# -----------------------------------------------------------------------------
# Edit this
# -----------------------------------------------------------------------------
RUN_DIRECTORY = Path("/scratch/planetseismology/mgclark/JVSRP/runs_paper/A14_Run15_LinearGradient_ExplosiveMomentTensor_NoInterface_Topography_NonlinearScattering_50pct_Compaction1over6")
OUTPUT_MP4 = Path(__file__).resolve().parent / f"{RUN_DIRECTORY.name}_shot_17_U3_two_sided_wavefield.mp4"

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
START_TIME = 0.0
END_TIME = 5.0
FRAME_DT = 0.01
FPS = 15
CHANNEL = "U3"
NX = 500
DPI = 240
IO_CHUNK_FRAMES = 25
COLOUR_PERCENTILE = 99.7
COLOUR_SCAN_FRAMES = 41
SOURCE_EXCLUSION_RADIUS = 2.0
SURFACE_PADDING = 2.0
FIGSIZE = (8.8, 6.8)

run_root = station_file = source_file = model_file = output_config_file = cache_file = None
output_mp4 = None
element_slices = []

def run_label(path):
    match = re.search(r'(?:^|_)Run[_-]?(\d+)(?:_|$)', Path(path).name, flags=re.I)
    return f"Run {match.group(1).zfill(2)}" if match else Path(path).name

def rank_number(path):
    match = re.search(r'\.rank(\d+)$', Path(path).name)
    if match is None:
        raise ValueError(f'Could not determine rank number from {path}')
    return int(match.group(1))

def find_rank_files(directory):
    return sorted([p for p in Path(directory).glob('axisem3d_synthetics.nc.rank*') if re.search(r'\.rank\d+$', p.name)], key=rank_number)

def axis_number(dimensions, word):
    matches = [i for i, name in enumerate(dimensions) if word.lower() in name.lower()]
    return matches[0] if len(matches) == 1 else None

def read_strings(variable):
    values = np.asarray(variable[:])
    if values.dtype.kind == 'S':
        if values.ndim >= 2 and values.dtype.itemsize == 1:
            values = values.reshape(-1, values.shape[-1])
            return [b''.join(row).decode('utf-8', errors='replace').replace('\x00', '').strip() for row in values]
        return [value.decode('utf-8', errors='replace').replace('\x00', '').strip() for value in values.reshape(-1)]
    if values.dtype.kind == 'U' and values.ndim >= 2:
        values = values.reshape(-1, values.shape[-1])
        return [''.join(row).replace('\x00', '').strip() for row in values]
    return [str(value).replace('\x00', '').strip() for value in values.reshape(-1)]

def resolve_input_file(path_value, description):
    path = Path(str(path_value)).expanduser()
    candidates = [path] if path.is_absolute() else [model_file.parent / path, run_root / path, run_root / 'input' / path]
    candidates += [model_file.parent / path.name, run_root / 'input' / path.name, run_root / path.name]
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Could not find the {description}. Checked:\n' + '\n'.join(map(str, checked)))

def configure_run_paths(run_directory):
    global run_root, station_file, source_file, model_file, output_config_file, cache_file, output_mp4, element_slices
    run_root = Path(run_directory).expanduser().resolve()
    source_file = run_root / 'input' / 'inparam.source.yaml'
    model_file = run_root / 'input' / 'inparam.model.yaml'
    output_config_file = run_root / 'input' / 'inparam.output.yaml'
    output_mp4 = OUTPUT_MP4.expanduser().resolve()
    cache_file = output_mp4.with_name(f'.{run_root.name}_wavefield_cache_{os.getpid()}.dat')
    with output_config_file.open(encoding='utf-8') as handle:
        document = yaml.safe_load(handle) or {}
    def groups(key):
        result = {}
        for item in document.get(key, []) or []:
            name, config = next(iter(item.items()))
            result[name] = config
        return result
    station_groups = groups('list_of_station_groups')
    station_file = resolve_input_file(station_groups['shot_17']['locations']['station_file'], 'Shot-17 station file')
    element_groups = groups('list_of_element_groups')
    element_slices = []
    for name, phi, sign in (('scattering_slice_south', math.pi, -1), ('scattering_slice_north', 0.0, 1)):
        config = element_groups[name]
        _, hi = map(float, config['elements']['horizontal_range'])
        directory = run_root / 'output' / 'elements' / name
        rank_files = find_rank_files(directory)
        if not rank_files:
            raise FileNotFoundError(f'No rank files found in {directory}')
        element_slices.append(dict(name=name, phi=phi, sign=sign, radial_max=hi, rank_files=rank_files))

def read_model_document():
    with model_file.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file) or {}

def standard_coordinates(dataset):
    variable = dataset.variables['list_element_coords']
    values = np.asarray(variable[:], dtype=np.float64)
    dimensions = list(variable.dimensions)
    while values.ndim > 3:
        singleton_axes = [i for i, n in enumerate(values.shape) if n == 1]
        if not singleton_axes:
            raise ValueError('list_element_coords has too many non-singleton dimensions')
        axis = singleton_axes[0]
        values = np.squeeze(values, axis=axis)
        dimensions.pop(axis)
    if values.ndim != 3:
        raise ValueError('list_element_coords is not three-dimensional')
    element_axis = axis_number(dimensions, 'element')
    gll_axis = axis_number(dimensions, 'gll')
    coordinate_axis = next((i for i, name in enumerate(dimensions) if 'coord' in name.lower()), None)
    if coordinate_axis is None:
        coordinate_axis = [i for i, n in enumerate(values.shape) if n == 2 and i not in (element_axis, gll_axis)][0]
    if element_axis is None:
        element_axis = [i for i in range(3) if i not in (coordinate_axis, gll_axis)][0]
    if gll_axis is None:
        gll_axis = [i for i in range(3) if i not in (coordinate_axis, element_axis)][0]
    return np.moveaxis(values, (element_axis, coordinate_axis, gll_axis), (0, 1, 2))

def map_element_ids(element_ids, table, number_of_coordinates):
    values = np.asarray(table)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != number_of_coordinates and values.shape[1] == number_of_coordinates:
        values = values.T
    if values.shape[0] != number_of_coordinates:
        return None
    element_ids = np.asarray(element_ids, dtype=np.int64)
    for column in range(values.shape[1]):
        candidate = np.asarray(values[:, column], dtype=np.int64)
        order = np.argsort(candidate, kind='mergesort')
        sorted_values = candidate[order]
        positions = np.searchsorted(sorted_values, element_ids)
        if np.all(positions < sorted_values.size) and np.all(sorted_values[positions] == element_ids):
            rows = order[positions]
            if np.unique(rows).size == rows.size:
                return rows
    return None


def coordinate_indices(dataset, wave_name, number_of_elements, number_of_coordinates):
    if number_of_elements == number_of_coordinates:
        return np.arange(number_of_elements, dtype=np.int64)
    suffix = wave_name[len('data_wave'):] if wave_name.startswith('data_wave') else ''
    mapping_name = 'list_element' + suffix
    if mapping_name not in dataset.variables:
        candidates = [name for name, variable in dataset.variables.items() if name.startswith('list_element') and name not in ('list_element_coords', 'list_element_na') and int(np.prod(variable.shape)) == number_of_elements]
        if len(candidates) != 1:
            raise ValueError(f'Could not map {wave_name} to element coordinates')
        mapping_name = candidates[0]
    element_ids = np.asarray(dataset.variables[mapping_name][:], dtype=np.int64).reshape(-1)
    if 'list_element_na' in dataset.variables:
        mapped = map_element_ids(element_ids, dataset.variables['list_element_na'][:], number_of_coordinates)
        if mapped is not None:
            return mapped
    if element_ids.min() >= 0 and element_ids.max() < number_of_coordinates:
        return element_ids
    if element_ids.min() >= 1 and element_ids.max() <= number_of_coordinates:
        return element_ids - 1
    raise ValueError(f'Could not map {wave_name} element IDs')

def read_exodus_mesh_geometry(model_document):
    mesh_path = resolve_input_file(model_document['model1D']['exodus_mesh'], 'Exodus mesh')
    with Dataset(mesh_path, mode='r') as dataset:
        dataset.set_auto_mask(False)
        raw_discontinuities = np.asarray(dataset.variables['discontinuities'][:], dtype=np.float64).reshape(-1) if 'discontinuities' in dataset.variables else np.array([], dtype=np.float64)
        if 'coordy' in dataset.variables:
            vertical = np.asarray(dataset.variables['coordy'][:], dtype=np.float64).reshape(-1)
        else:
            variable = dataset.variables['coord']
            coordinates = np.asarray(variable[:], dtype=np.float64)
            dimensions = list(variable.dimensions)
            coordinate_axis = next((i for i, name in enumerate(dimensions) if 'num_dim' in name.lower() or name.lower() in {'dimension', 'dimensions'}), None)
            if coordinate_axis is None:
                coordinate_axis = [i for i, n in enumerate(coordinates.shape) if n in (2, 3)][0]
            coordinates = np.moveaxis(coordinates, coordinate_axis, 0)
            vertical = coordinates[1].reshape(-1)
        horizontal = np.asarray(dataset.variables['coordx'][:], dtype=np.float64).reshape(-1) if 'coordx' in dataset.variables else np.array([], dtype=np.float64)
    vertical = vertical[np.isfinite(vertical)]
    surface_z = float(np.max(vertical))
    mesh_depth = surface_z - float(np.min(vertical))
    tol = max(0.1, 1e-3 * mesh_depth)
    mesh_boundaries = []
    for discontinuity_z in raw_discontinuities[np.isfinite(raw_discontinuities)]:
        depth = surface_z - float(discontinuity_z)
        if tol < depth < mesh_depth - tol:
            mesh_boundaries.append(depth)
    return {'path': mesh_path, 'surface_z': surface_z, 'depth': mesh_depth, 'boundaries': sorted({round(float(depth), 8) for depth in mesh_boundaries}), 'x_min': float(np.min(horizontal)) if horizontal.size else math.nan, 'x_max': float(np.max(horizontal)) if horizontal.size else math.nan}

def read_geometric_model_configurations(model_document):
    configurations = []
    for item in model_document.get('list_of_3D_models', []) or []:
        model_name, model = next(iter(item.items()))
        if not (isinstance(model, dict) and model.get('activated', False) and model.get('class_name') == 'StructuredGridG3D'):
            continue
        if 'undulation_range' not in model or 'undulation_data' not in model:
            continue
        coordinates = model.get('coordinates', {}) or {}
        coordinate_variables = coordinates.get('nc_variables', ['x', 'y'])
        grid_length_factor = 1.0 if str(coordinates.get('length_unit', 'm')).lower() == 'm' else 1000.0
        lower, upper = sorted((float(v) for v in model['undulation_range']['min_max']))
        configurations.append({'name': str(model_name), 'file': resolve_input_file(model['nc_data_file'], f'NetCDF file for geometric model {model_name!r}'), 'interface': float(model['undulation_range']['interface']), 'minimum': lower, 'maximum': upper, 'variable': str(model['undulation_data']['nc_var']), 'factor': float(model['undulation_data'].get('factor', 1.0)), 'x_variable': str(coordinate_variables[0]), 'y_variable': str(coordinate_variables[1]), 'grid_length_factor': grid_length_factor})
    return configurations

def standard_xy_array(dataset, variable_name, x_variable_name, y_variable_name, grid_length_factor):
    x = np.asarray(dataset.variables[x_variable_name][:], dtype=np.float64).reshape(-1) * grid_length_factor
    y = np.asarray(dataset.variables[y_variable_name][:], dtype=np.float64).reshape(-1) * grid_length_factor
    variable = dataset.variables[variable_name]
    values = np.asarray(variable[:], dtype=np.float64)
    dimensions = list(variable.dimensions)
    x_axis = dimensions.index(dataset.variables[x_variable_name].dimensions[0])
    y_axis = dimensions.index(dataset.variables[y_variable_name].dimensions[0])
    values = np.moveaxis(values, (x_axis, y_axis), (0, 1))
    while values.ndim > 2:
        singleton_axes = [axis for axis in range(2, values.ndim) if values.shape[axis] == 1]
        if not singleton_axes:
            raise ValueError(f'Geometric variable {variable_name!r} has extra non-singleton dimensions')
        values = np.squeeze(values, axis=singleton_axes[0])
    if x[0] > x[-1]:
        x, values = x[::-1], values[::-1, :]
    if y[0] > y[-1]:
        y, values = y[::-1], values[:, ::-1]
    return x, y, values

def bilinear_sample(x, y, values, query_x, query_y):
    query_x = np.asarray(query_x, dtype=np.float64)
    query_y = np.asarray(query_y, dtype=np.float64)
    query_x = np.clip(query_x, x[0], x[-1])
    query_y = np.clip(query_y, y[0], y[-1])
    ix = np.clip(np.searchsorted(x, query_x, side='right') - 1, 0, x.size - 2)
    iy = np.clip(np.searchsorted(y, query_y, side='right') - 1, 0, y.size - 2)
    x0, x1, y0, y1 = x[ix], x[ix + 1], y[iy], y[iy + 1]
    tx = np.divide(query_x - x0, x1 - x0, out=np.zeros_like(query_x), where=x1 != x0)
    ty = np.divide(query_y - y0, y1 - y0, out=np.zeros_like(query_y), where=y1 != y0)
    v00, v10, v01, v11 = values[ix, iy], values[ix + 1, iy], values[ix, iy + 1], values[ix + 1, iy + 1]
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11

def geometric_weight(reference_depth, configuration):
    lower, interface, upper = configuration['minimum'], configuration['interface'], configuration['maximum']
    depth = np.asarray(reference_depth, dtype=np.float64)
    weight = np.zeros_like(depth)
    left = (depth >= lower) & (depth <= interface)
    if interface > lower:
        weight[left] = (depth[left] - lower) / (interface - lower)
    right = (depth >= interface) & (depth <= upper)
    if upper > interface:
        weight[right] = np.maximum(weight[right], (upper - depth[right]) / (upper - interface))
    return np.clip(weight, 0.0, 1.0)

def load_geometric_models(configurations, distance, phi):
    query_x, query_y = distance * math.cos(phi), distance * math.sin(phi)
    models = []
    for configuration in configurations:
        with Dataset(configuration['file'], mode='r') as dataset:
            dataset.set_auto_mask(False)
            x_grid, y_grid, values = standard_xy_array(dataset, configuration['variable'], configuration['x_variable'], configuration['y_variable'], configuration['grid_length_factor'])
        values *= configuration['factor']
        profile = bilinear_sample(x_grid, y_grid, values, query_x, query_y)
        model = dict(configuration)
        model.update({'x_grid': x_grid, 'y_grid': y_grid, 'values': values, 'profile': profile})
        models.append(model)
    return models

def sample_geometric_model(model, query_x, query_y):
    return float(bilinear_sample(model['x_grid'], model['y_grid'], model['values'], np.array([query_x]), np.array([query_y]))[0])

def point_geometric_displacement(models, query_x, query_y, reference_depth):
    displacement = 0.0
    for model in models:
        weight = float(geometric_weight(np.array([reference_depth]), model)[0])
        if weight:
            displacement += weight * sample_geometric_model(model, query_x, query_y)
    return displacement


def deformed_depth_curve(reference_depth, geometry):
    curve = np.full_like(geometry['distance'], float(reference_depth), dtype=np.float64)
    for model in geometry['models']:
        weight = float(geometric_weight(np.array([reference_depth]), model)[0])
        if weight:
            curve -= weight * model['profile']
    return curve

def read_source_depth():
    with source_file.open('r', encoding='utf-8') as file:
        document = yaml.safe_load(file)
    source = next(iter(document['list_of_sources'][0].values()))
    return float(source['location']['depth'])

def read_stations():
    stations = {}
    with station_file.open(encoding='utf-8') as handle:
        for line in handle:
            values = line.split('#', 1)[0].split()
            if not values or values[0] not in {'G1', 'G2', 'G3'}:
                continue
            name = values[0]
            radius, phi, depth = float(values[2]), float(values[3]), float(values[5])
            stations[name] = dict(distance=radius, azimuth=phi, depth=depth, x=radius * math.cos(phi), y=radius * math.sin(phi))
    expected = {'G3': -73.152, 'G2': -27.432, 'G1': 18.288}
    for name, x in expected.items():
        if abs(stations[name]['x'] - x) > 0.01:
            raise ValueError(f'{name} is at x={stations[name]["x"]:.3f} m; Shot 17 expects {x:+.3f} m')
    return stations

def find_bm_file():
    input_dir = run_root / 'input'
    referenced_names = []
    mesh_yaml = input_dir / 'input.mesh.yaml'
    if mesh_yaml.is_file():
        with mesh_yaml.open('r', encoding='utf-8') as file:
            mesh_document = yaml.safe_load(file) or {}
        model_value = (mesh_document.get('basic', {}) or {}).get('model')
        if model_value and str(model_value).lower().endswith('.bm'):
            referenced_names.append(Path(str(model_value)).expanduser())
    if not referenced_names:
        for yaml_file in input_dir.rglob('*.yaml'):
            uncommented = [line.split('#', 1)[0] for line in yaml_file.read_text(encoding='utf-8', errors='replace').splitlines()]
            referenced_names.extend(Path(name).expanduser() for name in re.findall(r'[^\s"\'=:,\[\]{}]+\.bm', '\n'.join(uncommented)))
    found = []
    for name in referenced_names:
        for candidate in [name, input_dir / name, run_root / name, input_dir / name.name, run_root / name.name, input_dir / 'vel_models' / name.name, run_root / 'vel_models' / name.name, run_root.parent.parent / 'vel_models' / name.name]:
            if candidate.is_file():
                found.append(candidate.resolve())
    found = sorted(set(found))
    if len(found) == 1:
        return found[0]
    if len(found) > 1 and len({path.name for path in found}) == 1:
        return found[0]
    if found:
        raise ValueError('More than one different .bm file is referenced by the run:\n' + '\n'.join(map(str, found)))
    local_candidates = sorted(set(run_root.rglob('*.bm')))
    if len(local_candidates) == 1:
        return local_candidates[0].resolve()
    raise FileNotFoundError('Could not locate the .bm velocity model used by this run')

def read_bm(path):
    length_factor = velocity_factor = 1000.0
    rows = []
    with Path(path).open('r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            upper = line.upper()
            if upper.startswith('UNITS'):
                units = line.split()[1].lower()
                length_factor = velocity_factor = 1.0 if units == 'm' else 1000.0
                continue
            if upper.startswith(('NAME', 'ANELASTIC', 'ANISOTROPIC', 'COLUMNS')):
                continue
            values = line.split()
            if len(values) < 3:
                continue
            numeric = tuple(float(v) for v in values)
            rows.append((numeric[0] * length_factor, numeric[1] * velocity_factor, numeric[2] * velocity_factor, numeric[1:]))
    boundaries = []
    for i in range(1, len(rows)):
        if np.isclose(rows[i][0], rows[i - 1][0]) and rows[i][3] != rows[i - 1][3]:
            boundaries.append(float(rows[i][0]))
    return {'boundaries': sorted(set(boundaries))}

def scan_wavefield(slices):
    blocks, reference_time = [], None
    z_min, z_max = math.inf, -math.inf
    for spec in slices:
        radial_min, radial_max = math.inf, -math.inf
        side_z_min, side_z_max = math.inf, -math.inf
        for rank_file in spec['rank_files']:
            with Dataset(rank_file, mode='r') as dataset:
                dataset.set_auto_mask(False)
                time = np.asarray(dataset.variables['data_time'][:], dtype=np.float64).reshape(-1)
                if reference_time is None:
                    reference_time = time
                elif time.shape != reference_time.shape or not np.allclose(time, reference_time, rtol=1e-10, atol=1e-12):
                    raise ValueError(f'Time axis differs between saved slices/ranks: {rank_file}')
                channels = read_strings(dataset.variables['list_channel'])
                coordinates = standard_coordinates(dataset)
                for wave_name, variable in dataset.variables.items():
                    if not wave_name.startswith('data_'):
                        continue
                    dimensions, shape = tuple(variable.dimensions), tuple(variable.shape)
                    axes = [axis_number(dimensions, word) for word in ('time', 'element', 'channel', 'gll')]
                    if any(axis is None for axis in axes):
                        continue
                    time_axis, element_axis, channel_axis, gll_axis = axes
                    integer_axes = {channel_axis: channels.index(CHANNEL)}
                    for axis in range(len(shape)):
                        if axis not in axes:
                            integer_axes[axis] = 0
                    rows = coordinate_indices(dataset, wave_name, shape[element_axis], coordinates.shape[0])
                    xyz = coordinates[rows]
                    radial_min = min(radial_min, float(xyz[:, 0, :].min()))
                    radial_max = max(radial_max, float(xyz[:, 0, :].max()))
                    side_z_min = min(side_z_min, float(xyz[:, 1, :].min()))
                    side_z_max = max(side_z_max, float(xyz[:, 1, :].max()))
                    blocks.append(dict(id=len(blocks), path=rank_file, wave_name=wave_name, dimensions=dimensions, time_axis=time_axis, element_axis=element_axis, gll_axis=gll_axis, integer_axes=integer_axes, number_of_gll=shape[gll_axis], coordinate_indices=rows, side_sign=spec['sign']))
        if radial_min < -1e-5 or radial_min > 0.1 or radial_max < spec['radial_max'] - 0.2:
            raise ValueError(f'{spec["name"]} radial coverage [{radial_min:g}, {radial_max:g}] does not cover [0, {spec["radial_max"]:g}] m')
        z_min, z_max = min(z_min, side_z_min), max(z_max, side_z_max)
    limits = [spec['sign'] * spec['radial_max'] for spec in slices]
    return dict(blocks=blocks, time=reference_time, x_min=min(limits), x_max=max(limits), z_min=z_min, z_max=z_max)

def choose_frames(raw_time):
    target_time = np.arange(max(START_TIME, float(raw_time[0])), min(END_TIME, float(raw_time[-1])) + 0.25 * FRAME_DT, FRAME_DT)
    right = np.clip(np.searchsorted(raw_time, target_time, side='left'), 0, len(raw_time) - 1)
    left = np.clip(right - 1, 0, len(raw_time) - 1)
    use_left = np.abs(raw_time[left] - target_time) <= np.abs(raw_time[right] - target_time)
    indices = np.where(use_left, left, right).astype(np.int64)
    keep = np.r_[True, indices[1:] != indices[:-1]]
    return indices[keep], raw_time[indices[keep]]


def interpolation_weights(points, queries):
    triangulation = Delaunay(points)
    simplex = triangulation.find_simplex(queries, tol=1e-10)
    valid = simplex >= 0
    selected = simplex[valid]
    transform = triangulation.transform[selected]
    first_two = np.einsum('ijk,ik->ij', transform[:, :2, :], queries[valid] - transform[:, 2, :])
    weights = np.column_stack((first_two, 1.0 - first_two.sum(axis=1)))
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return valid, triangulation.simplices[selected], weights

def build_mapping(scan, mesh_geometry):
    surface_z, mesh_depth = mesh_geometry['surface_z'], mesh_geometry['depth']
    x_min, x_max = scan['x_min'], scan['x_max']
    depth_min = max(0.0, surface_z - scan['z_max'])
    depth_max = min(mesh_depth, surface_z - scan['z_min'])
    ny = max(2, int(round(NX * (depth_max - depth_min) / (x_max - x_min))))
    grid_x = np.linspace(x_min, x_max, NX)
    grid_depth = np.linspace(depth_min, depth_max, ny)
    xx, dd = np.meshgrid(grid_x, grid_depth)
    pixel_points = np.column_stack((xx.reshape(-1), dd.reshape(-1)))
    valid_mask = np.zeros(NX * ny, dtype=bool)
    blocks_by_file, blocks_by_id = defaultdict(list), {}
    side_parts = {-1: [], 1: []}
    for block in scan['blocks']:
        blocks_by_file[block['path']].append(block)
        blocks_by_id[block['id']] = block
        block['selected_pixels'] = np.empty(0, dtype=np.int64)
        block['selected_local'] = np.empty(0, dtype=np.int64)
        block['interpolation_weights'] = None
    for rank_file in sorted(blocks_by_file, key=lambda path: (rank_number(path), str(path))):
        with Dataset(rank_file, mode='r') as dataset:
            dataset.set_auto_mask(False)
            coordinates = standard_coordinates(dataset)
            for block in blocks_by_file[rank_file]:
                coords = coordinates[block['coordinate_indices']]
                signed_x = coords[:, 0, :].reshape(-1) * block['side_sign']
                depths = surface_z - coords[:, 1, :].reshape(-1)
                points = np.column_stack((signed_x, depths))
                side_parts[block['side_sign']].append((points, np.full(len(points), block['id'], dtype=np.int64), np.arange(len(points), dtype=np.int64)))
    for side in (-1, 1):
        points = np.concatenate([part[0] for part in side_parts[side]])
        owners = np.concatenate([part[1] for part in side_parts[side]])
        local = np.concatenate([part[2] for part in side_parts[side]])
        _, unique = np.unique(np.round(points, decimals=8), axis=0, return_index=True)
        points, owners, local = points[unique], owners[unique], local[unique]
        pixels = np.flatnonzero(xx.reshape(-1) < 0 if side < 0 else xx.reshape(-1) >= 0)
        covered, vertices, weights = interpolation_weights(points, pixel_points[pixels])
        covered_pixels = pixels[covered]
        valid_mask[covered_pixels] = True
        contribution_pixels = np.repeat(covered_pixels, 3)
        contribution_owner = owners[vertices].reshape(-1)
        contribution_local = local[vertices].reshape(-1)
        contribution_weights = weights.reshape(-1)
        used = contribution_weights > 0.0
        contribution_pixels = contribution_pixels[used]
        contribution_owner = contribution_owner[used]
        contribution_local = contribution_local[used]
        contribution_weights = contribution_weights[used]
        order = np.argsort(contribution_owner, kind='stable')
        owner_sorted = contribution_owner[order]
        splits = np.r_[0, np.flatnonzero(np.diff(owner_sorted)) + 1, len(order)]
        for first, last in zip(splits[:-1], splits[1:]):
            selected = order[first:last]
            if not len(selected):
                continue
            block = blocks_by_id[int(owner_sorted[first])]
            block_pixels, row = np.unique(contribution_pixels[selected], return_inverse=True)
            block_local, column = np.unique(contribution_local[selected], return_inverse=True)
            block['selected_pixels'] = block_pixels
            block['selected_local'] = block_local
            block['interpolation_weights'] = csr_matrix((contribution_weights[selected], (row, column)), shape=(len(block_pixels), len(block_local)))
    return dict(x_min=x_min, x_max=x_max, depth_min=depth_min, depth_max=depth_max, surface_z=surface_z, nx=NX, ny=ny, shape=(ny, NX), number_of_pixels=NX * ny, valid_mask=valid_mask.reshape(ny, NX))


def invalid_sample_masks(variable, values):
    masks = {'NaN': np.isnan(values), 'infinity': np.isinf(values)}
    markers = []
    for attribute in ('_FillValue', 'missing_value'):
        if attribute in variable.ncattrs():
            markers.extend((attribute, value) for value in np.asarray(variable.getncattr(attribute)).reshape(-1))
    if '_FillValue' not in variable.ncattrs():
        dtype = np.dtype(variable.dtype)
        default_key = f'{dtype.kind}{dtype.itemsize}'
        if default_key in default_fillvals:
            markers.append(('default NetCDF fill', default_fillvals[default_key]))
    for name, marker in markers:
        if np.isfinite(marker):
            masks[name] = masks.get(name, np.zeros(values.shape, dtype=bool)) | (values == marker)
    return masks

def read_wave_chunk(dataset, block, raw_indices):
    variable = dataset.variables[block['wave_name']]
    selection = [slice(None)] * len(block['dimensions'])
    for axis, index in block['integer_axes'].items():
        selection[axis] = int(index)
    raw_start, raw_stop = int(raw_indices[0]), int(raw_indices[-1]) + 1
    selection[block['time_axis']] = slice(raw_start, raw_stop)
    values = np.asarray(variable[tuple(selection)])
    remaining_axes = [axis for axis in range(len(block['dimensions'])) if axis not in block['integer_axes']]
    values = np.moveaxis(values, (remaining_axes.index(block['time_axis']), remaining_axes.index(block['element_axis']), remaining_axes.index(block['gll_axis'])), (0, 1, 2))
    values = values[raw_indices - raw_start].reshape(len(raw_indices), -1)
    used = values[:, block['selected_local']]
    masks = invalid_sample_masks(variable, used)
    if any(mask.any() for mask in masks.values()):
        raise ValueError('Invalid sample used by the movie')
    return values

def build_cache(scan, mapping, raw_indices):
    cache = np.memmap(cache_file, mode='w+', dtype=np.float64, shape=(len(raw_indices), mapping['number_of_pixels']))
    cache[:] = 0.0
    blocks_by_file = defaultdict(list)
    for block in scan['blocks']:
        blocks_by_file[block['path']].append(block)
    for rank_file in sorted(blocks_by_file, key=lambda path: (rank_number(path), str(path))):
        with Dataset(rank_file, mode='r') as dataset:
            dataset.set_auto_mask(False)
            for block in blocks_by_file[rank_file]:
                pixels = block['selected_pixels']
                if not pixels.size:
                    continue
                weights = block['interpolation_weights']
                for first in range(0, len(raw_indices), IO_CHUNK_FRAMES):
                    last = min(len(raw_indices), first + IO_CHUNK_FRAMES)
                    values = read_wave_chunk(dataset, block, raw_indices[first:last])
                    selected = values[:, block['selected_local']]
                    contribution = (weights @ selected.T).T if weights is not None else selected
                    cache[first:last, pixels] += contribution
        cache.flush()
    missing = ~np.asarray(mapping['valid_mask']).reshape(-1)
    cache[:, missing] = np.nan
    cache.flush()
    return cache

def display_frame(cache, frame_number, mapping):
    values = np.asarray(cache[frame_number], dtype=np.float64).reshape(mapping['shape'])
    return np.where(mapping['valid_mask'], values, np.nan)

def colour_limit(cache, mapping, source_depth):
    x = np.linspace(mapping['x_min'], mapping['x_max'], mapping['nx'])
    depth = np.linspace(mapping['depth_min'], mapping['depth_max'], mapping['ny'])
    xx, dd = np.meshgrid(x, depth)
    use_pixel = np.hypot(xx, dd - source_depth) >= SOURCE_EXCLUSION_RADIUS
    frame_numbers = np.unique(np.linspace(0, cache.shape[0] - 1, min(COLOUR_SCAN_FRAMES, cache.shape[0])).astype(int))
    limits = []
    for frame_number in frame_numbers:
        values = np.abs(display_frame(cache, frame_number, mapping)[use_pixel])
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            limits.append(float(np.percentile(values, COLOUR_PERCENTILE)))
    return float(np.percentile(limits, 90.0))

def build_display_geometry(mapping, stations, source_depth, model_document):
    distance = np.linspace(mapping['x_min'], mapping['x_max'], mapping['nx'])
    reference_depth = np.linspace(mapping['depth_min'], mapping['depth_max'], mapping['ny'])
    models = load_geometric_models(read_geometric_model_configurations(model_document), distance, 0.0)
    display_depth = np.broadcast_to(reference_depth[:, None], (mapping['ny'], mapping['nx'])).copy()
    for model in models:
        display_depth -= geometric_weight(reference_depth, model)[:, None] * model['profile'][None, :]
    geometry = {'distance': distance, 'reference_depth': reference_depth, 'display_x': np.broadcast_to(distance[None, :], display_depth.shape), 'display_depth': display_depth, 'models': models}
    surface_depth = deformed_depth_curve(0.0, geometry)
    geometry.update({'surface_depth': surface_depth, 'source_display_depth': source_depth - point_geometric_displacement(models, 0.0, 0.0, source_depth), 'station_display_depth': {name: station['depth'] - point_geometric_displacement(models, station['x'], station['y'], station['depth']) for name, station in stations.items()}})
    return geometry

def collect_boundary_specs(model, geometry, mapping, mesh_geometry):
    boundaries = []
    mesh_boundaries = mesh_geometry.get('boundaries', [])
    for depth in sorted(set(float(v) for v in model['boundaries'])):
        if not mapping['depth_min'] < depth < mapping['depth_max']:
            continue
        active_geometry = [item['name'] for item in geometry['models'] if float(geometric_weight(np.array([depth]), item)[0]) != 0.0 and np.any(np.abs(item['profile']) > 1e-10)]
        boundaries.append(dict(depth=depth, geometry_names=active_geometry, in_exodus=any(abs(depth - item) < 0.1 for item in mesh_boundaries)))
    return boundaries


def write_mp4(frames_per_second, draw_frame):
    figure = draw_frame('setup')
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    process = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(frames_per_second), '-i', '-', '-an',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_mp4),
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_number in range(draw_frame('count')):
        figure = draw_frame(frame_number)
        figure.canvas.draw()
        process.stdin.write(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode('utf-8', errors='replace')
    return_code = process.wait()
    plt.close(figure)
    if return_code != 0:
        raise RuntimeError(f'FFmpeg exited with code {return_code}:\n{stderr}')

def make_movie(cache, mapping, frame_times, stations, boundary_specs, display_geometry, vmax, title):
    figure = plt.figure(figsize=FIGSIZE, dpi=DPI)
    wave_axis = figure.add_axes([0.095, 0.105, 0.77, 0.745])
    colourbar_axis = figure.add_axes([0.884, 0.105, 0.022, 0.745])
    first_frame = np.clip(display_frame(cache, 0, mapping) / vmax, -1.0, 1.0)
    image = wave_axis.pcolormesh(display_geometry['display_x'], display_geometry['display_depth'], first_frame, cmap='RdBu_r', vmin=-1.0, vmax=1.0, shading='nearest', rasterized=True, zorder=1)
    surface_depth = display_geometry['surface_depth']
    top_limit = float(np.nanmin(surface_depth)) - SURFACE_PADDING
    wave_axis.fill_between(display_geometry['distance'], top_limit, surface_depth, facecolor='white', edgecolor='none', zorder=4)
    wave_axis.plot(display_geometry['distance'], surface_depth, color='black', linewidth=1.35, zorder=5)
    wave_axis.set_xlim(mapping['x_min'], mapping['x_max'])
    wave_axis.set_ylim(max(mapping['depth_max'], float(np.nanmax(display_geometry['display_depth']))), top_limit)
    wave_axis.set_xlabel('Distance from Shot 17: south (-) / north (+)')
    wave_axis.set_ylabel('Depth below reference surface (m)')
    wave_axis.spines['top'].set_visible(False)
    colourbar = figure.colorbar(image, cax=colourbar_axis)
    colourbar.set_label('Normalised vertical displacement')
    colourbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    for boundary in boundary_specs:
        curve = deformed_depth_curve(boundary['depth'], display_geometry)
        is_geometric = bool(boundary['geometry_names'])
        wave_axis.plot(display_geometry['distance'], curve, color='0.10' if is_geometric else '0.28', linestyle='--' if is_geometric else ':', linewidth=1.45 if is_geometric else 1.1, alpha=0.98 if is_geometric else 0.88, zorder=6 if is_geometric else 5)
        wave_axis.annotate(f'{boundary["depth"]:g} m reference interface', xy=(display_geometry['distance'][0], curve[0]), xytext=(5, -4), textcoords='offset points', fontsize=7.5, ha='left', va='top', color='0.10', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.7, 'pad': 1.0}, zorder=7)
    wave_axis.scatter(0.0, display_geometry['source_display_depth'], marker='*', s=115, facecolor='gold', edgecolor='black', linewidth=0.7, zorder=8)
    for name in ('G3', 'G2', 'G1'):
        station = stations[name]
        station_depth = display_geometry['station_display_depth'][name]
        wave_axis.scatter(station['x'], station_depth, marker='v', s=62, facecolor='white', edgecolor='black', linewidth=0.9, zorder=50, clip_on=False)
        wave_axis.annotate(name, xy=(station['x'], station_depth), xytext=(0, 7), textcoords='offset points', fontsize=8.5, fontweight='bold', ha='center', va='bottom', clip_on=False, zorder=50)
    time_text = wave_axis.text(0.982, 0.022, f't = {frame_times[0]:.3f} s', transform=wave_axis.transAxes, ha='right', va='bottom', fontsize=11.5, fontweight='bold', clip_on=False, bbox={'boxstyle': 'round,pad=0.28', 'facecolor': 'white', 'edgecolor': '0.55', 'linewidth': 0.6, 'alpha': 0.96}, zorder=10000)
    figure.suptitle(title, x=0.5, y=0.955, fontsize=11.5, fontweight='bold')

    def draw_frame(which):
        if which == 'setup':
            return figure
        if which == 'count':
            return len(frame_times)
        frame = np.clip(display_frame(cache, which, mapping) / vmax, -1.0, 1.0)
        image.set_array(frame.reshape(-1))
        time_text.set_text(f't = {frame_times[which]:.3f} s')
        return figure

    write_mp4(FPS, draw_frame)

def main():
    configure_run_paths(RUN_DIRECTORY)
    OUTPUT_MP4.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    model_document = read_model_document()
    mesh_geometry = read_exodus_mesh_geometry(model_document)
    model = read_bm(find_bm_file())
    source_depth = read_source_depth()
    stations = read_stations()
    scan = scan_wavefield(element_slices)
    raw_indices, frame_times = choose_frames(scan['time'])
    mapping = build_mapping(scan, mesh_geometry)
    display_geometry = build_display_geometry(mapping, stations, source_depth, model_document)
    boundary_specs = collect_boundary_specs(model, display_geometry, mapping, mesh_geometry)
    cache = None
    try:
        cache = build_cache(scan, mapping, raw_indices)
        vmax = colour_limit(cache, mapping, source_depth)
        make_movie(cache, mapping, frame_times, stations, boundary_specs, display_geometry, vmax, run_label(run_root))
    finally:
        if cache is not None:
            del cache
        if cache_file is not None and Path(cache_file).exists():
            Path(cache_file).unlink()

if __name__ == '__main__':
    main()
