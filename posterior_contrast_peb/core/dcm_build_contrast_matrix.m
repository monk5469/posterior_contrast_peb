function [L, info] = dcm_build_contrast_matrix(Pnames, contrast_spec)
%DCM_BUILD_CONTRAST_MATRIX Build a posterior-contrast matrix from Pnames.
%
%   [L, INFO] = DCM_BUILD_CONTRAST_MATRIX(PNAMES, CONTRAST_SPEC)
%
%   PNAMES can be a cell array of SPM-style parameter names, e.g.
%   {'B(1,1,1)', 'B(2,1,1)', 'B(1,1,2)', 'B(2,1,2)'}.
%
%   CONTRAST_SPEC.contrasts(k).terms is a struct array. Each term should
%   include:
%       weight      numeric scalar
%       field       e.g. 'B'
%       row/target/to or col/source/from
%       condition   numeric index or label from condition_labels
%
%   Rows are target regions and columns are source regions, following SPM's
%   matrix indexing convention for A/B matrices.

    if nargin < 2
        error('dcm_build_contrast_matrix:NotEnoughInputs', ...
            'Pnames and contrast_spec are required.');
    end

    Pnames = normalize_pnames(Pnames);
    n_params = numel(Pnames);
    param_info = parse_pnames(Pnames);

    if ~isfield(contrast_spec, 'contrasts')
        error('dcm_build_contrast_matrix:MissingContrasts', ...
            'contrast_spec.contrasts is required.');
    end

    contrasts = contrast_spec.contrasts;
    if ~isstruct(contrasts)
        error('dcm_build_contrast_matrix:InvalidContrasts', ...
            'contrast_spec.contrasts must be a struct array.');
    end

    n_contrasts = numel(contrasts);
    L = zeros(n_contrasts, n_params);
    rows = repmat(struct('name', '', 'terms', []), n_contrasts, 1);

    for c = 1:n_contrasts
        if isfield(contrasts(c), 'name')
            rows(c).name = contrasts(c).name;
        else
            rows(c).name = sprintf('contrast_%d', c);
        end

        if ~isfield(contrasts(c), 'terms') || isempty(contrasts(c).terms)
            error('dcm_build_contrast_matrix:MissingTerms', ...
                'Contrast %d has no terms.', c);
        end

        terms = contrasts(c).terms;
        resolved_terms = repmat(struct( ...
            'weight', [], 'field', '', 'row', [], 'col', [], ...
            'condition', [], 'pname', '', 'index', []), numel(terms), 1);

        for t = 1:numel(terms)
            term = terms(t);
            weight = get_optional(term, 'weight', 1);
            field_name = upper(get_optional(term, 'field', get_optional(contrast_spec, 'field', 'B')));
            row_idx = resolve_region_index(term, contrast_spec, {'row', 'target', 'to'});
            col_idx = resolve_region_index(term, contrast_spec, {'col', 'source', 'from'});
            cond_idx = resolve_condition_index(term, contrast_spec);

            pidx = find_param(param_info, field_name, row_idx, col_idx, cond_idx);
            if numel(pidx) ~= 1
                error('dcm_build_contrast_matrix:ParameterMatchFailed', ...
                    'Expected one match for term %d in contrast "%s"; found %d.', ...
                    t, rows(c).name, numel(pidx));
            end

            L(c, pidx) = L(c, pidx) + weight;

            resolved_terms(t).weight = weight;
            resolved_terms(t).field = field_name;
            resolved_terms(t).row = row_idx;
            resolved_terms(t).col = col_idx;
            resolved_terms(t).condition = cond_idx;
            resolved_terms(t).pname = Pnames{pidx};
            resolved_terms(t).index = pidx;
        end

        rows(c).terms = resolved_terms;
    end

    info = struct();
    info.Pnames = Pnames;
    info.parameters = param_info;
    info.contrasts = rows;
end

function Pnames = normalize_pnames(Pnames)
    if ischar(Pnames)
        Pnames = cellstr(Pnames);
    end
    if isstring(Pnames)
        Pnames = cellstr(Pnames);
    end
    if ~iscell(Pnames)
        error('dcm_build_contrast_matrix:InvalidPnames', ...
            'Pnames must be a cell array, char array, or string array.');
    end
    Pnames = Pnames(:)';
end

function param_info = parse_pnames(Pnames)
    n = numel(Pnames);
    param_info = repmat(struct('name', '', 'field', '', 'row', [], ...
        'col', [], 'condition', [], 'indices', []), n, 1);

    for i = 1:n
        name = strtrim(Pnames{i});
        tokens = regexp(name, '^([A-Za-z]+)\(([^)]*)\)', 'tokens', 'once');
        if isempty(tokens)
            error('dcm_build_contrast_matrix:CannotParsePname', ...
                'Cannot parse parameter name: %s', name);
        end

        field_name = upper(tokens{1});
        nums = regexp(tokens{2}, '\d+', 'match');
        idx = str2double(nums);

        param_info(i).name = name;
        param_info(i).field = field_name;
        param_info(i).indices = idx(:)';
        if numel(idx) >= 1
            param_info(i).row = idx(1);
        end
        if numel(idx) >= 2
            param_info(i).col = idx(2);
        end
        if numel(idx) >= 3
            param_info(i).condition = idx(3);
        end
    end
end

function idx = find_param(param_info, field_name, row_idx, col_idx, cond_idx)
    mask = false(numel(param_info), 1);
    for i = 1:numel(param_info)
        ok = strcmpi(param_info(i).field, field_name);
        if ~isempty(row_idx)
            ok = ok && isequal(param_info(i).row, row_idx);
        end
        if ~isempty(col_idx)
            ok = ok && isequal(param_info(i).col, col_idx);
        end
        if ~isempty(cond_idx)
            ok = ok && isequal(param_info(i).condition, cond_idx);
        end
        mask(i) = ok;
    end
    idx = find(mask);
end

function value = get_optional(s, field_name, default_value)
    if isstruct(s) && isfield(s, field_name) && ~isempty(s.(field_name))
        value = s.(field_name);
    else
        value = default_value;
    end
end

function idx = resolve_region_index(term, contrast_spec, aliases)
    idx = [];
    for a = 1:numel(aliases)
        name = aliases{a};
        if isfield(term, name) && ~isempty(term.(name))
            value = term.(name);
            idx = resolve_label_or_index(value, get_optional(contrast_spec, 'region_labels', {}));
            return;
        end
    end
end

function idx = resolve_condition_index(term, contrast_spec)
    if isfield(term, 'condition') && ~isempty(term.condition)
        idx = resolve_label_or_index(term.condition, get_optional(contrast_spec, 'condition_labels', {}));
    elseif isfield(term, 'condition_index') && ~isempty(term.condition_index)
        idx = term.condition_index;
    else
        idx = [];
    end
end

function idx = resolve_label_or_index(value, labels)
    if isnumeric(value)
        idx = value;
        return;
    end
    if isstring(value)
        value = char(value);
    end
    if ~ischar(value)
        error('dcm_build_contrast_matrix:InvalidLabel', ...
            'Labels must be char, string, or numeric indices.');
    end
    if isempty(labels)
        error('dcm_build_contrast_matrix:MissingLabels', ...
            'Label "%s" was provided but no labels were supplied.', value);
    end
    if isstring(labels)
        labels = cellstr(labels);
    end
    match = find(strcmp(labels, value));
    if numel(match) ~= 1
        error('dcm_build_contrast_matrix:LabelNotFound', ...
            'Expected one match for label "%s"; found %d.', value, numel(match));
    end
    idx = match;
end

