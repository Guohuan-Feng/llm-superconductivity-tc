"""Fixed 127-column Tc classification input recipe, root selected 2026-10-09.

No target, label, model, imputation, fitting, or file I/O. Actual geometrical
distances are recalculated from fractional coordinates, cell vectors and images.
The caller must authenticate original TRAIN identities and the geometry source.
"""
from __future__ import annotations
import math
import numpy as np

COMPOSITION_COLUMNS=tuple(f'occupancy_fraction_Z{z:03d}' for z in range(1,119))
GEOMETRY_COLUMNS=(
    'volume_per_occupied_atom_A3',
    'nearest_positive_PBC_distance_site_occ_mean_A',
    'nearest_positive_PBC_distance_site_occ_std_A',
    'nearest_positive_PBC_distance_min_A',
    'nearest_positive_PBC_distance_max_A',
    'coordination6A_site_occ_mean',
    'coordination6A_site_occ_std',
    'PBC_edge_distance_pair_occ_mean_A',
    'PBC_edge_distance_pair_occ_std_A',
)
ALL_COLUMNS=COMPOSITION_COLUMNS+GEOMETRY_COLUMNS
RECIPE_VERSION='root_fixed_20261009_118occupancy_9geometry_v1'
NEIGHBOR_CUTOFF_A=6.0
NEIGHBOR_BOUND_TOLERANCE_A=1e-8
COORDINATE_CACHE_CHECK_TOLERANCE_A=1e-7

class StructureInputError(ValueError):
    pass

def require(condition,code):
    if not bool(condition):
        raise StructureInputError(code)

def _weighted_mean_std(values,weights):
    values=np.asarray(values,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    require(values.ndim==weights.ndim==1 and len(values)==len(weights)>0,'weighted_shape')
    require(np.isfinite(values).all() and np.isfinite(weights).all(),'weighted_nonfinite')
    require((weights>0).all(),'weighted_nonpositive_mass')
    mass=math.fsum(float(x) for x in weights)
    require(math.isfinite(mass) and mass>0,'weighted_mass')
    mean=math.fsum(float(w)*float(v) for w,v in zip(weights,values))/mass
    variance=math.fsum(float(w)*(float(v)-mean)**2 for w,v in zip(weights,values))/mass
    require(math.isfinite(mean) and math.isfinite(variance) and variance>=0,'weighted_overflow')
    return mean,math.sqrt(variance)

def features(record):
    """Return composition118, geometry9 and quantitative source-check witnesses.

    Population standard deviations. Partial occupancy remains an absolute
    occupancy, not a guessed ordered species. Nonzero periodic self images count.
    The cache's 6 A edge completeness is inherited from its authenticated
    pymatgen generator; this function verifies each edge rather than rediscovering
    all periodic image edges or claiming a fresh CIF symmetry expansion.
    """
    require(not {'tc','class','label','target','target_K','y'}.intersection(record),'outcome_key_forbidden')
    n=record.get('n_sites')
    require(isinstance(n,(int,np.integer)) and not isinstance(n,bool) and n>0,'n_sites')
    frac=np.asarray(record['frac_coords'],dtype=np.float64)
    lattice=np.asarray(record['lattice'],dtype=np.float64)
    site_occ=np.asarray(record['site_occupancy'],dtype=np.float64)
    require(frac.shape==(n,3) and lattice.shape==(3,3) and site_occ.shape==(n,),'coordinate_shape')
    require(np.isfinite(frac).all() and np.isfinite(lattice).all() and np.isfinite(site_occ).all(),'coordinate_nonfinite')
    require((site_occ>=0).all() and (site_occ>0).any(),'site_occupancy')
    volume=abs(float(np.linalg.det(lattice)))
    require(math.isfinite(volume) and volume>0,'degenerate_cell')
    recorded_volume=float(record['volume'])
    require(math.isfinite(recorded_volume) and recorded_volume>0,'cached_volume')
    require(abs(volume-recorded_volume)<=1e-6*max(1.0,volume),'cached_volume_mismatch')

    raw_species_site=np.asarray(record['species_site'])
    raw_z=np.asarray(record['species_Z'])
    species_occ=np.asarray(record['species_occupancy'],dtype=np.float64)
    require(raw_species_site.ndim==raw_z.ndim==species_occ.ndim==1,'species_shape')
    require(len(raw_species_site)==len(raw_z)==len(species_occ)>0,'species_length')
    require(np.issubdtype(raw_species_site.dtype,np.integer) and np.issubdtype(raw_z.dtype,np.integer),'species_index_not_integer')
    species_site=raw_species_site.astype(np.int64,copy=False)
    z=raw_z.astype(np.int64,copy=False)
    require(((species_site>=0)&(species_site<n)).all() and ((z>=1)&(z<=118)).all(),'species_index_bounds')
    require(np.isfinite(species_occ).all() and (species_occ>=0).all(),'species_occupancy')
    reconstructed_occ=np.bincount(species_site,weights=species_occ,minlength=n)
    require(np.allclose(reconstructed_occ,site_occ,rtol=0,atol=1e-12),'species_site_occupancy_mismatch')
    total_occ=math.fsum(float(x) for x in site_occ)
    require(math.isfinite(total_occ) and total_occ>0,'occupied_atom_mass')
    composition=np.zeros(118,dtype=np.float64)
    for atomic_number in range(1,119):
        composition[atomic_number-1]=math.fsum(float(x) for x in species_occ[z==atomic_number])/total_occ
    require(np.isfinite(composition).all() and (composition>=0).all(),'composition_nonfinite')
    require(abs(math.fsum(float(x) for x in composition)-1.0)<=1e-12,'composition_not_normalized')

    raw_centers=np.asarray(record['neighbor_center'])
    raw_indices=np.asarray(record['neighbor_index'])
    raw_images=np.asarray(record['neighbor_image'])
    cached_distances=np.asarray(record['neighbor_distance'],dtype=np.float64)
    require(raw_centers.ndim==raw_indices.ndim==cached_distances.ndim==1,'neighbor_shape')
    require(len(raw_centers)==len(raw_indices)==len(cached_distances)>0,'neighbor_empty')
    require(raw_images.shape==(len(raw_indices),3),'neighbor_image_shape')
    require(np.issubdtype(raw_centers.dtype,np.integer) and np.issubdtype(raw_indices.dtype,np.integer)
            and np.issubdtype(raw_images.dtype,np.integer),'neighbor_indices_not_integer')
    center=raw_centers.astype(np.int64,copy=False)
    index=raw_indices.astype(np.int64,copy=False)
    images=raw_images.astype(np.int64,copy=False)
    require(((center>=0)&(center<n)&(index>=0)&(index<n)).all(),'neighbor_index_bounds')
    require(not ((center==index)&np.all(images==0,axis=1)).any(),'zero_image_self_edge')
    distances=np.linalg.norm((frac[index]+images-frac[center])@lattice,axis=1)
    require(np.isfinite(distances).all() and (distances>0).all(),'neighbor_nonpositive_or_nonfinite')
    require((distances<=NEIGHBOR_CUTOFF_A+NEIGHBOR_BOUND_TOLERANCE_A).all(),'neighbor_outside_cutoff')
    require(np.isfinite(cached_distances).all(),'cached_neighbor_nonfinite')
    max_cache_diff=float(np.max(np.abs(distances-cached_distances)))
    require(max_cache_diff<=COORDINATE_CACHE_CHECK_TOLERANCE_A,'cached_neighbor_mismatch')
    positive=(site_occ>0)
    included=positive[center]&positive[index]
    center=center[included]; index=index[included]; distances=distances[included]
    require(len(distances)>0,'no_positive_occupied_edges')
    nearest=np.full(n,np.inf,dtype=np.float64)
    np.minimum.at(nearest,center,distances)
    require(np.isfinite(nearest[positive]).all(),'positive_site_without_neighbor')
    matrix=np.asarray(record['distance_matrix'],dtype=np.float64)
    require(matrix.shape==(n,n) and np.isfinite(matrix).all() and (matrix>0).all(),'cached_periodic_matrix')
    matrix_nearest=np.min(matrix[np.ix_(positive,positive)],axis=1)
    nearest_cache_diff=float(np.max(np.abs(nearest[positive]-matrix_nearest)))
    require(nearest_cache_diff<=COORDINATE_CACHE_CHECK_TOLERANCE_A,'nearest_cache_crosscheck_mismatch')
    site_weights=site_occ[positive]
    nn_mean,nn_std=_weighted_mean_std(nearest[positive],site_weights)
    coordination=np.bincount(center,weights=site_occ[index],minlength=n)
    coordination_mean,coordination_std=_weighted_mean_std(coordination[positive],site_weights)
    pair_weights=site_occ[center]*site_occ[index]
    edge_mean,edge_std=_weighted_mean_std(distances,pair_weights)
    geometry=np.asarray([volume/total_occ,nn_mean,nn_std,float(np.min(nearest[positive])),
                         float(np.max(nearest[positive])),coordination_mean,coordination_std,
                         edge_mean,edge_std],dtype=np.float64)
    require(geometry.shape==(9,) and np.isfinite(geometry).all(),'geometry_nonfinite')
    witnesses={'n_sites':int(n),'positive_occupied_sites':int(positive.sum()),'occupied_atom_mass':total_occ,
               'cached_PBC_edges':len(cached_distances),'positive_occupied_PBC_edges':len(distances),
               'nonzero_self_image_edges':int(np.sum(center==index)),
               'neighbor_coordinate_cache_max_abs_difference_A':max_cache_diff,
               'nearest_matrix_edge_crosscheck_max_abs_difference_A':nearest_cache_diff,
               'volume_cache_abs_difference_A3':abs(volume-recorded_volume)}
    return composition,geometry,witnesses

def feature_vector(record):
    composition,geometry,witnesses=features(record)
    return np.concatenate([composition,geometry]),witnesses
