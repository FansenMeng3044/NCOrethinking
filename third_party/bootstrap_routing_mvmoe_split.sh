#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
upstream_dir=${1:-"${script_dir}/Routing-MVMoE"}
overlay_dir="${script_dir}/Routing-MVMoE-overlay"
upstream_url=https://github.com/RoyalSkye/Routing-MVMoE.git
upstream_commit=af29e5af0595f94f3ecc3bc46d72df1089a62682

if [[ ! -d "${overlay_dir}" ]]; then
    echo "Missing tracked overlay: ${overlay_dir}" >&2
    exit 1
fi

if [[ ! -d "${upstream_dir}/.git" ]]; then
    git clone "${upstream_url}" "${upstream_dir}"
fi

if [[ -n "$(git -C "${upstream_dir}" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to overwrite tracked changes in ${upstream_dir}" >&2
    exit 1
fi

if [[ "$(git -C "${upstream_dir}" rev-parse HEAD)" != "${upstream_commit}" ]]; then
    git -C "${upstream_dir}" fetch origin "${upstream_commit}"
    git -C "${upstream_dir}" checkout --detach "${upstream_commit}"
fi

cp -a "${overlay_dir}/." "${upstream_dir}/"
python -m py_compile \
    "${upstream_dir}/train_split.py" \
    "${upstream_dir}/SplitTrainer.py" \
    "${upstream_dir}/training_metrics.py" \
    "${upstream_dir}/split_models/xy_models.py"

printf '%s\n' "${upstream_dir}"
