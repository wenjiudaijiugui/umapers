use crate::{Metric, UmapError};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

type KnnRows = (Vec<Vec<usize>>, Vec<Vec<f32>>);

#[derive(Debug, Clone)]
pub struct SparseCsrMatrix {
    n_rows: usize,
    n_cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
    squared_norms: Vec<f32>,
}

impl SparseCsrMatrix {
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<Self, UmapError> {
        if n_cols == 0 {
            return Err(UmapError::InvalidParameter(
                "csr n_cols must be >= 1".to_string(),
            ));
        }
        if indptr.len() != n_rows + 1 {
            return Err(UmapError::InvalidParameter(format!(
                "csr indptr length must be n_rows + 1 (got {}, expected {})",
                indptr.len(),
                n_rows + 1
            )));
        }
        if indptr.first().copied().unwrap_or_default() != 0 {
            return Err(UmapError::InvalidParameter(
                "csr indptr must start from 0".to_string(),
            ));
        }
        if indices.len() != data.len() {
            return Err(UmapError::InvalidParameter(
                "csr indices/data lengths must match".to_string(),
            ));
        }
        if indptr[n_rows] != indices.len() {
            return Err(UmapError::InvalidParameter(format!(
                "csr indptr last value ({}) must equal nnz ({})",
                indptr[n_rows],
                indices.len()
            )));
        }

        for row in 0..n_rows {
            if indptr[row] > indptr[row + 1] {
                return Err(UmapError::InvalidParameter(format!(
                    "csr indptr must be non-decreasing, got indptr[{row}]={} > indptr[{}]={}",
                    indptr[row],
                    row + 1,
                    indptr[row + 1]
                )));
            }
            let start = indptr[row];
            let end = indptr[row + 1];
            let row_indices = &indices[start..end];
            for w in row_indices.windows(2) {
                if w[0] >= w[1] {
                    return Err(UmapError::InvalidParameter(format!(
                        "csr row {row} indices must be strictly increasing"
                    )));
                }
            }
            for &col in row_indices {
                if col >= n_cols {
                    return Err(UmapError::InvalidParameter(format!(
                        "csr row {row} has column index {col} out of bounds for n_cols={n_cols}"
                    )));
                }
            }
        }

        if data.iter().any(|v| !v.is_finite()) {
            return Err(UmapError::InvalidParameter(
                "csr data must be finite".to_string(),
            ));
        }

        let mut squared_norms = vec![0.0_f32; n_rows];
        for (row, norm_slot) in squared_norms.iter_mut().enumerate() {
            let (_, vals) = row_slices(&indptr, &indices, &data, row);
            *norm_slot = vals.iter().map(|v| v * v).sum();
        }

        Ok(Self {
            n_rows,
            n_cols,
            indptr,
            indices,
            data,
            squared_norms,
        })
    }

    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn squared_norm(&self, row: usize) -> f32 {
        self.squared_norms[row]
    }

    #[inline]
    pub fn row(&self, row: usize) -> (&[usize], &[f32]) {
        row_slices(&self.indptr, &self.indices, &self.data, row)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct NeighborCandidate {
    idx: usize,
    dist: f32,
}

impl Eq for NeighborCandidate {}

impl Ord for NeighborCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for NeighborCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn is_better(a: NeighborCandidate, b: NeighborCandidate) -> bool {
    match a.dist.total_cmp(&b.dist) {
        Ordering::Less => true,
        Ordering::Equal => a.idx < b.idx,
        Ordering::Greater => false,
    }
}

#[inline]
fn push_top_k(heap: &mut BinaryHeap<NeighborCandidate>, cand: NeighborCandidate, k: usize) {
    if heap.len() < k {
        heap.push(cand);
        return;
    }
    if let Some(&worst) = heap.peek()
        && is_better(cand, worst)
    {
        heap.pop();
        heap.push(cand);
    }
}

#[inline]
fn heap_into_sorted_rows(heap: BinaryHeap<NeighborCandidate>) -> (Vec<usize>, Vec<f32>) {
    let mut row = heap.into_vec();
    row.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.idx.cmp(&b.idx)));
    let mut idx = Vec::with_capacity(row.len());
    let mut dist = Vec::with_capacity(row.len());
    for item in row {
        idx.push(item.idx);
        dist.push(item.dist);
    }
    (idx, dist)
}

#[inline]
fn row_slices<'a>(
    indptr: &[usize],
    indices: &'a [usize],
    data: &'a [f32],
    row: usize,
) -> (&'a [usize], &'a [f32]) {
    let start = indptr[row];
    let end = indptr[row + 1];
    (&indices[start..end], &data[start..end])
}

#[inline]
fn sparse_dot(lhs_idx: &[usize], lhs_vals: &[f32], rhs_idx: &[usize], rhs_vals: &[f32]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut dot = 0.0_f32;

    while i < lhs_idx.len() && j < rhs_idx.len() {
        match lhs_idx[i].cmp(&rhs_idx[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                dot += lhs_vals[i] * rhs_vals[j];
                i += 1;
                j += 1;
            }
        }
    }

    dot
}

#[inline]
fn euclidean_distance_from_dot(norm_lhs: f32, norm_rhs: f32, dot: f32) -> f32 {
    (norm_lhs + norm_rhs - 2.0 * dot).max(0.0).sqrt()
}

#[inline]
fn sparse_row_euclidean_distance(matrix: &SparseCsrMatrix, lhs: usize, rhs: usize) -> f32 {
    let (lhs_idx, lhs_vals) = matrix.row(lhs);
    let (rhs_idx, rhs_vals) = matrix.row(rhs);
    let dot = sparse_dot(lhs_idx, lhs_vals, rhs_idx, rhs_vals);
    euclidean_distance_from_dot(matrix.squared_norm(lhs), matrix.squared_norm(rhs), dot)
}

#[inline]
fn sparse_row_manhattan_distance(matrix: &SparseCsrMatrix, lhs: usize, rhs: usize) -> f32 {
    let (lhs_idx, lhs_vals) = matrix.row(lhs);
    let (rhs_idx, rhs_vals) = matrix.row(rhs);
    let mut i = 0usize;
    let mut j = 0usize;
    let mut sum = 0.0_f32;

    while i < lhs_idx.len() && j < rhs_idx.len() {
        match lhs_idx[i].cmp(&rhs_idx[j]) {
            Ordering::Less => {
                sum += lhs_vals[i].abs();
                i += 1;
            }
            Ordering::Greater => {
                sum += rhs_vals[j].abs();
                j += 1;
            }
            Ordering::Equal => {
                sum += (lhs_vals[i] - rhs_vals[j]).abs();
                i += 1;
                j += 1;
            }
        }
    }

    while i < lhs_idx.len() {
        sum += lhs_vals[i].abs();
        i += 1;
    }
    while j < rhs_idx.len() {
        sum += rhs_vals[j].abs();
        j += 1;
    }

    sum
}

#[inline]
fn sparse_row_cosine_distance(matrix: &SparseCsrMatrix, lhs: usize, rhs: usize) -> f32 {
    let (lhs_idx, lhs_vals) = matrix.row(lhs);
    let (rhs_idx, rhs_vals) = matrix.row(rhs);
    let dot = sparse_dot(lhs_idx, lhs_vals, rhs_idx, rhs_vals);
    let lhs_norm = matrix.squared_norm(lhs).sqrt();
    let rhs_norm = matrix.squared_norm(rhs).sqrt();

    if lhs_norm == 0.0 && rhs_norm == 0.0 {
        0.0
    } else if lhs_norm == 0.0 || rhs_norm == 0.0 {
        1.0
    } else {
        let cosine_sim = (dot / (lhs_norm * rhs_norm)).clamp(-1.0, 1.0);
        1.0 - cosine_sim
    }
}

#[inline]
fn sparse_row_distance(matrix: &SparseCsrMatrix, lhs: usize, rhs: usize, metric: Metric) -> f32 {
    match metric {
        Metric::Euclidean => sparse_row_euclidean_distance(matrix, lhs, rhs),
        Metric::Manhattan => sparse_row_manhattan_distance(matrix, lhs, rhs),
        Metric::Cosine => sparse_row_cosine_distance(matrix, lhs, rhs),
    }
}

pub(crate) fn exact_nearest_neighbors(
    data: &SparseCsrMatrix,
    n_neighbors: usize,
    metric: Metric,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.n_rows();
    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut heap = BinaryHeap::<NeighborCandidate>::with_capacity(n_neighbors + 1);
            for j in 0..n_samples {
                let dist = if i == j {
                    0.0
                } else {
                    sparse_row_distance(data, i, j, metric)
                };
                push_top_k(&mut heap, NeighborCandidate { idx: j, dist }, n_neighbors);
            }
            heap_into_sorted_rows(heap)
        })
        .unzip()
}

fn dense_query_to_sparse_manhattan_distance(
    query_row: &[f32],
    query_l1_norm: f32,
    sparse_cols: &[usize],
    sparse_vals: &[f32],
) -> f32 {
    let mut sum = query_l1_norm;
    for (&col, &val) in sparse_cols.iter().zip(sparse_vals.iter()) {
        let qv = query_row[col];
        sum += (qv - val).abs() - qv.abs();
    }
    sum
}

fn dense_query_to_sparse_cosine_distance(
    query_row: &[f32],
    query_l2_norm: f32,
    sparse_cols: &[usize],
    sparse_vals: &[f32],
    sparse_l2_norm: f32,
) -> f32 {
    if query_l2_norm == 0.0 && sparse_l2_norm == 0.0 {
        return 0.0;
    }
    if query_l2_norm == 0.0 || sparse_l2_norm == 0.0 {
        return 1.0;
    }

    let mut dot = 0.0_f32;
    for (&col, &val) in sparse_cols.iter().zip(sparse_vals.iter()) {
        dot += query_row[col] * val;
    }
    let cosine_sim = (dot / (query_l2_norm * sparse_l2_norm)).clamp(-1.0, 1.0);
    1.0 - cosine_sim
}

pub(crate) fn exact_nearest_neighbors_dense_query(
    query: &[Vec<f32>],
    reference: &SparseCsrMatrix,
    n_neighbors: usize,
    metric: Metric,
) -> Result<KnnRows, UmapError> {
    let n_features = reference.n_cols();
    for (row, q) in query.iter().enumerate() {
        if q.len() != n_features {
            if row == 0 {
                return Err(UmapError::FeatureMismatch {
                    expected: n_features,
                    got: q.len(),
                });
            }
            return Err(UmapError::InconsistentDimensions {
                row,
                expected: n_features,
                got: q.len(),
            });
        }
    }

    let query_sq_norms = if matches!(metric, Metric::Euclidean | Metric::Cosine) {
        Some(
            query
                .iter()
                .map(|row| row.iter().map(|v| v * v).sum::<f32>())
                .collect::<Vec<f32>>(),
        )
    } else {
        None
    };
    let query_l1_norms = if matches!(metric, Metric::Manhattan) {
        Some(
            query
                .iter()
                .map(|row| row.iter().map(|v| v.abs()).sum::<f32>())
                .collect::<Vec<f32>>(),
        )
    } else {
        None
    };
    let query_l2_norms = if matches!(metric, Metric::Cosine) {
        Some(
            query_sq_norms
                .as_ref()
                .expect("cosine query norms should be computed")
                .iter()
                .map(|norm| norm.sqrt())
                .collect::<Vec<f32>>(),
        )
    } else {
        None
    };
    let reference_l2_norms = if matches!(metric, Metric::Cosine) {
        Some(
            (0..reference.n_rows())
                .map(|row| reference.squared_norm(row).sqrt())
                .collect::<Vec<f32>>(),
        )
    } else {
        None
    };

    let mut indices = Vec::with_capacity(query.len());
    let mut dists = Vec::with_capacity(query.len());
    for (q_row_idx, q_row) in query.iter().enumerate() {
        let mut heap = BinaryHeap::<NeighborCandidate>::with_capacity(n_neighbors + 1);
        for ref_idx in 0..reference.n_rows() {
            let (ref_cols, ref_vals) = reference.row(ref_idx);
            let dist = match metric {
                Metric::Euclidean => {
                    let mut dot = 0.0_f32;
                    for (&col, &val) in ref_cols.iter().zip(ref_vals.iter()) {
                        dot += q_row[col] * val;
                    }
                    euclidean_distance_from_dot(
                        query_sq_norms
                            .as_ref()
                            .expect("euclidean query norms should be computed")[q_row_idx],
                        reference.squared_norm(ref_idx),
                        dot,
                    )
                }
                Metric::Manhattan => dense_query_to_sparse_manhattan_distance(
                    q_row,
                    query_l1_norms
                        .as_ref()
                        .expect("manhattan query norms should be computed")[q_row_idx],
                    ref_cols,
                    ref_vals,
                ),
                Metric::Cosine => dense_query_to_sparse_cosine_distance(
                    q_row,
                    query_l2_norms
                        .as_ref()
                        .expect("cosine query norms should be computed")[q_row_idx],
                    ref_cols,
                    ref_vals,
                    reference_l2_norms
                        .as_ref()
                        .expect("cosine reference norms should be computed")[ref_idx],
                ),
            };
            push_top_k(
                &mut heap,
                NeighborCandidate { idx: ref_idx, dist },
                n_neighbors,
            );
        }

        let (idx_row, dist_row) = heap_into_sorted_rows(heap);
        indices.push(idx_row);
        dists.push(dist_row);
    }

    Ok((indices, dists))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_data() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 2.0, 0.0, 0.0],
            vec![0.0, 2.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    }

    fn dense_to_csr(data: &[Vec<f32>]) -> SparseCsrMatrix {
        let n_rows = data.len();
        let n_cols = data[0].len();
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.push(0);
        for row in data {
            for (col, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    indices.push(col);
                    values.push(val);
                }
            }
            indptr.push(indices.len());
        }
        SparseCsrMatrix::new(n_rows, n_cols, indptr, indices, values).expect("valid csr")
    }

    fn dense_distance(x: &[f32], y: &[f32], metric: Metric) -> f32 {
        match metric {
            Metric::Euclidean => x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f32>()
                .sqrt(),
            Metric::Manhattan => x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum(),
            Metric::Cosine => {
                let dot = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>();
                let nx = x.iter().map(|v| v * v).sum::<f32>().sqrt();
                let ny = y.iter().map(|v| v * v).sum::<f32>().sqrt();
                if nx == 0.0 && ny == 0.0 {
                    0.0
                } else if nx == 0.0 || ny == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (nx * ny)).clamp(-1.0, 1.0)
                }
            }
        }
    }

    fn brute_knn(data: &[Vec<f32>], k: usize, metric: Metric) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let n = data.len();
        let mut idx = vec![vec![0usize; k]; n];
        let mut dist = vec![vec![0.0f32; k]; n];
        for i in 0..n {
            let mut row = (0..n)
                .map(|j| (j, dense_distance(&data[i], &data[j], metric)))
                .collect::<Vec<_>>();
            row.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            for kk in 0..k {
                idx[i][kk] = row[kk].0;
                dist[i][kk] = row[kk].1;
            }
        }
        (idx, dist)
    }

    #[test]
    fn sparse_knn_matches_dense_for_all_supported_metrics() {
        let data = dense_data();
        let csr = dense_to_csr(&data);
        let k = 3;

        for metric in [Metric::Euclidean, Metric::Manhattan, Metric::Cosine] {
            let (s_idx, s_dist) = exact_nearest_neighbors(&csr, k, metric);
            let (d_idx, d_dist) = brute_knn(&data, k, metric);
            assert_eq!(s_idx, d_idx);
            for (sr, dr) in s_dist.iter().zip(d_dist.iter()) {
                for (&lhs, &rhs) in sr.iter().zip(dr.iter()) {
                    assert!(
                        (lhs - rhs).abs() <= 1e-6,
                        "distance mismatch {lhs} vs {rhs}"
                    );
                }
            }
        }
    }

    #[test]
    fn dense_query_to_sparse_knn_matches_bruteforce_for_all_supported_metrics() {
        let reference = dense_data();
        let csr = dense_to_csr(&reference);
        let query = vec![vec![0.5, 0.0, 1.5, 0.0, 0.0], vec![0.0, 1.0, 0.0, 1.0, 0.0]];
        let k = 2;

        for metric in [Metric::Euclidean, Metric::Manhattan, Metric::Cosine] {
            let (idx, dists) = exact_nearest_neighbors_dense_query(&query, &csr, k, metric)
                .expect("query knn should succeed");
            for qi in 0..query.len() {
                let mut row = reference
                    .iter()
                    .enumerate()
                    .map(|(ri, r)| (ri, dense_distance(&query[qi], r, metric)))
                    .collect::<Vec<_>>();
                row.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                for kk in 0..k {
                    assert_eq!(idx[qi][kk], row[kk].0);
                    assert!((dists[qi][kk] - row[kk].1).abs() <= 1e-6);
                }
            }
        }
    }

    #[test]
    fn sparse_knn_and_dense_query_tie_break_is_deterministic() {
        let reference = vec![
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, -1.0],
        ];
        let csr = dense_to_csr(&reference);
        let query = vec![vec![0.0, 0.0]];
        let k = 3;

        for metric in [Metric::Euclidean, Metric::Manhattan, Metric::Cosine] {
            let (idx_a, dist_a) = exact_nearest_neighbors(&csr, k, metric);
            let (idx_b, dist_b) = exact_nearest_neighbors(&csr, k, metric);
            assert_eq!(idx_a, idx_b, "sparse self-knn index order should be stable");
            assert_eq!(
                dist_a, dist_b,
                "sparse self-knn distance order should be stable"
            );

            let (q_idx_a, q_dist_a) =
                exact_nearest_neighbors_dense_query(&query, &csr, k, metric).expect("query knn");
            let (q_idx_b, q_dist_b) =
                exact_nearest_neighbors_dense_query(&query, &csr, k, metric).expect("query knn");
            assert_eq!(
                q_idx_a, q_idx_b,
                "dense-query-to-sparse index order should be stable"
            );
            assert_eq!(
                q_dist_a, q_dist_b,
                "dense-query-to-sparse distance order should be stable"
            );

            let mut expected = reference
                .iter()
                .enumerate()
                .map(|(ri, r)| (ri, dense_distance(&query[0], r, metric)))
                .collect::<Vec<_>>();
            expected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            assert_eq!(
                q_idx_a[0],
                expected.iter().take(k).map(|x| x.0).collect::<Vec<_>>()
            );
        }
    }
}
