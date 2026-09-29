function export_empirical_release_components(secondary_file, finger_file, ...
    faces_input_file, geometry_input_file, finger_input_file, output_dir)
% Export non-identifying parameter orderings and fitted precision components.

if nargin ~= 6
    error(['Expected secondary MAT, FingerData MAT, three input MAT files, ' ...
        'and output directory.']);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

secondary = load(secondary_file, 'fits');
finger = load(finger_file, 'fits');
faces_input = load(faces_input_file, 'Pnames');
geometry_input = load(geometry_input_file, 'Pnames');
finger_input = load(finger_input_file, 'Pnames');
source_orderings = struct( ...
    'FacesData', {normalize_names(faces_input.Pnames)}, ...
    'GeometryData', {normalize_names(geometry_input.Pnames)}, ...
    'FingerData', {normalize_names(finger_input.Pnames)});

parameter_rows = struct([]);
component_rows = struct([]);
projection_rows = struct([]);
baseline_rows = struct([]);

datasets = fieldnames(source_orderings);
for dataset_index = 1:numel(datasets)
    dataset = datasets{dataset_index};
    names = source_orderings.(dataset);
    for source_index = 1:numel(names)
        row = struct( ...
            'dataset', dataset, ...
            'source_index', source_index, ...
            'parameter_name', names{source_index});
        parameter_rows = append_row(parameter_rows, row);
    end
end

[parameter_rows, component_rows, projection_rows, baseline_rows] = append_archive( ...
    parameter_rows, component_rows, projection_rows, baseline_rows, ...
    secondary.fits, source_orderings);
[parameter_rows, component_rows, projection_rows, baseline_rows] = append_archive( ...
    parameter_rows, component_rows, projection_rows, baseline_rows, ...
    finger.fits, source_orderings);

parameter_table = struct2table(parameter_rows);
parameter_table = unique(parameter_table, 'rows', 'stable');
parameter_table = sortrows(parameter_table, {'dataset', 'source_index'});
writetable(parameter_table, fullfile(output_dir, 'empirical_parameter_ordering.csv'));

component_table = struct2table(component_rows);
component_table = sortrows(component_table, ...
    {'dataset', 'target', 'specification', 'component_index', 'row_index', 'column_index'});
writetable(component_table, ...
    fullfile(output_dir, 'empirical_precision_components_long.csv'));

projection_table = struct2table(projection_rows);
projection_table = sortrows(projection_table, ...
    {'dataset', 'target', 'specification', 'target_row', 'coordinate_index'});
writetable(projection_table, ...
    fullfile(output_dir, 'empirical_fitted_target_projections.csv'));

baseline_table = struct2table(baseline_rows);
baseline_table = sortrows(baseline_table, ...
    {'dataset', 'target', 'specification', 'row_index', 'column_index'});
writetable(baseline_table, ...
    fullfile(output_dir, 'empirical_baseline_precision_long.csv'));
end


function [parameter_rows, component_rows, projection_rows, baseline_rows] = append_archive( ...
    parameter_rows, component_rows, projection_rows, baseline_rows, ...
    fits, source_orderings)
fit_names = fieldnames(fits);
for fit_index = 1:numel(fit_names)
    fit_name = fit_names{fit_index};
    fit = fits.(fit_name);
    [dataset, target, specification] = parse_fit_name(fit_name);
    peb = fit.PEB;
    source_names = source_orderings.(dataset);

    target_dimension = size(fit.target_projection, 1);
    coordinate_dimension = size(peb.M.Q{1}, 1);
    coordinate_system = coordinate_system_for(specification);
    coordinate_names = release_coordinate_names( ...
        coordinate_system, coordinate_dimension, target_dimension, source_names);
    log_scales = full(peb.Eh(:));

    weighted_precision = zeros(coordinate_dimension);
    for component_index = 1:numel(peb.M.Q)
        weighted_precision = weighted_precision + ...
            exp(log_scales(component_index)) * full(peb.M.Q{component_index});
    end
    fitted_precision = stable_inverse_release(full(peb.Ce));
    baseline_precision = (fitted_precision - weighted_precision);
    baseline_precision = (baseline_precision + baseline_precision') / 2;
    [baseline_row_indices, baseline_column_indices, baseline_values] = ...
        find(baseline_precision);
    for value_index = 1:numel(baseline_values)
        row_index = baseline_row_indices(value_index);
        column_index = baseline_column_indices(value_index);
        row = struct( ...
            'dataset', dataset, ...
            'target', target, ...
            'specification', specification, ...
            'coordinate_system', coordinate_system, ...
            'coordinate_dimension', coordinate_dimension, ...
            'row_index', row_index, ...
            'column_index', column_index, ...
            'row_name', coordinate_names{row_index}, ...
            'column_name', coordinate_names{column_index}, ...
            'value', baseline_values(value_index));
        baseline_rows = append_row(baseline_rows, row);
    end

    projection = full(fit.target_projection);
    [target_rows, coordinate_indices, projection_values] = find(projection);
    for projection_index = 1:numel(projection_values)
        coordinate_index = coordinate_indices(projection_index);
        row = struct( ...
            'dataset', dataset, ...
            'target', target, ...
            'specification', specification, ...
            'coordinate_system', coordinate_system, ...
            'target_dimension', target_dimension, ...
            'coordinate_dimension', coordinate_dimension, ...
            'target_row', target_rows(projection_index), ...
            'coordinate_index', coordinate_index, ...
            'coordinate_name', coordinate_names{coordinate_index}, ...
            'weight', projection_values(projection_index));
        projection_rows = append_row(projection_rows, row);
    end

    for component_index = 1:numel(peb.M.Q)
        component = full(peb.M.Q{component_index});
        [row_indices, column_indices, values] = find(component);
        scale = exp(log_scales(component_index));
        for value_index = 1:numel(values)
            row_index = row_indices(value_index);
            column_index = column_indices(value_index);
            row = struct( ...
                'dataset', dataset, ...
                'target', target, ...
                'specification', specification, ...
                'coordinate_system', coordinate_system, ...
                'target_dimension', target_dimension, ...
                'coordinate_dimension', coordinate_dimension, ...
                'component_index', component_index, ...
                'component_log_scale', log_scales(component_index), ...
                'component_scale', scale, ...
                'row_index', row_index, ...
                'column_index', column_index, ...
                'row_name', coordinate_names{row_index}, ...
                'column_name', coordinate_names{column_index}, ...
                'base_value', values(value_index), ...
                'weighted_value', scale * values(value_index));
            component_rows = append_row(component_rows, row);
        end
    end
end
end


function matrix = stable_inverse_release(matrix)
matrix = (matrix + matrix') / 2;
[vectors, values_matrix] = eig(matrix);
values = real(diag(values_matrix));
floor_value = max(max(abs(values)) * 1e-12, eps);
values(values < floor_value) = floor_value;
matrix = vectors * diag(1 ./ values) * vectors';
matrix = (matrix + matrix') / 2;
end


function [dataset, target, specification] = parse_fit_name(fit_name)
parts = strsplit(fit_name, '__');
if numel(parts) ~= 2
    error('Unexpected fit name: %s', fit_name);
end
prefix = parts{1};
specification = canonical_specification(parts{2});
if strcmp(prefix, 'FacesData')
    dataset = 'FacesData';
    target = 'rOFA_to_rFFA_unfamiliar_immediate_minus_long';
elseif strcmp(prefix, 'GeometryData')
    dataset = 'GeometryData';
    target = 'rOcc_to_rPar_regular_minus_random';
elseif strcmp(prefix, 'R1_bilateral_average')
    dataset = 'FingerData';
    target = 'R1_bilateral_average';
elseif strcmp(prefix, 'R2_left_right')
    dataset = 'FingerData';
    target = 'R2_left_right';
else
    error('Unexpected fit prefix: %s', prefix);
end
end


function specification = canonical_specification(value)
specification = value;
end


function value = coordinate_system_for(specification)
if any(strcmp(specification, {'source_single', 'source_all'}))
    value = 'source';
elseif strcmp(specification, 'target_direct')
    value = 'target';
else
    value = 'rotated_target_complement';
end
end


function names = release_coordinate_names(system, dimension, target_dimension, source_names)
if strcmp(system, 'source')
    names = source_names;
elseif strcmp(system, 'target')
    names = arrayfun(@(index) sprintf('target_%d', index), ...
        1:dimension, 'UniformOutput', false);
else
    names = cell(1, dimension);
    for index = 1:dimension
        if index <= target_dimension
            names{index} = sprintf('target_%d', index);
        else
            names{index} = sprintf('complement_%d', index - target_dimension);
        end
    end
end
if numel(names) ~= dimension
    error('Coordinate-name count does not match component dimension.');
end
end


function names = normalize_names(values)
if ischar(values)
    names = cellstr(values);
elseif isstring(values)
    names = cellstr(values(:));
elseif iscell(values)
    names = cellfun(@char, values(:), 'UniformOutput', false);
else
    error('Unsupported parameter-name representation: %s', class(values));
end
names = names(:)';
end


function rows = append_row(rows, row)
if isempty(rows)
    rows = row;
else
    rows(end + 1) = row;
end
end
