#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cholmod.h>

// There's a bug in the original MATLAB implementation that we are currently emulating.
#define BUG_FOR_BUG_COMPATABILITY_MODE

typedef long Int;
#define TRUE 1
#define FALSE 0
#define SPUMONI 1
#define NATURAL_ORDER 0
#define LOWER_TRIANGULAR -1

static PyObject * cholmod_lchol_c(PyObject *self, PyObject *args);

static PyMethodDef CholmodMethods[] = {
    {"lchol_c",  cholmod_lchol_c, METH_VARARGS,
     "LL^T factorization."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

void sputil_trim(cholmod_sparse *S, Int k, cholmod_common *cm)
{
    Int *Sp ;
    Int ncol ;
    size_t n1, nznew ;

    if (S == NULL)
    {
        return ;
    }

    ncol = S->ncol ;
    if (k < 0 || k >= ncol)
    {
        /* do not modify S */
        return ;
    }

    /* reduce S->p in size.  This cannot fail. */
    n1 = ncol + 1 ;
    S->p = cholmod_l_realloc (k+1, sizeof (Int), S->p, &n1, cm) ;

    /* get the new number of entries in S */
    Sp = S->p ;
    nznew = Sp [k] ;

    /* reduce S->i, S->x, and S->z (if present) to size nznew */
    cholmod_l_reallocate_sparse (nznew, S, cm) ;

    /* S now has only k columns */
    S->ncol = k ;
}

// adapted from cholmod matlab
void sputil_config (cholmod_common *cm)
{
    /* cholmod_l_solve must return a real or zomplex X for MATLAB */
    cm->prefer_zomplex = FALSE ;

#if (CHOLMOD_VERSION < CHOLMOD_VER_CODE(3,0))
    /* use mxMalloc and related memory management routines */
    cm->malloc_memory  = malloc;
    cm->free_memory    = free;
    cm->realloc_memory = realloc;
    cm->calloc_memory  = calloc;
    cm->print_function = printf;// NULL;
    /* complex arithmetic */
    cm->complex_divide = cholmod_l_divcomplex ;
    cm->hypotenuse     = cholmod_l_hypot ;
#endif

    cm->print = (int)SPUMONI + 2 ; // -1;

    /* error handler */
    cm->error_handler  = NULL;

#ifndef NPARTITION
#if defined(METIS_VERSION)
#if (METIS_VERSION >= METIS_VER(4,0,2))
    /* METIS 4.0.2 uses function pointers for malloc and free */
    METIS_malloc = cm->malloc_memory ;
    METIS_free   = cm->free_memory ;
#endif
#endif
#endif

    /* Turn off METIS memory guard.  It is not needed, because mxMalloc will
     * safely terminate the mexFunction and free any workspace without killing
     * all of MATLAB.  This assumes cholmod_make was used to compile CHOLMOD
     * for MATLAB. */
    cm->metis_memory = 0.0 ;

    cm->nmethods = 2;
    /* convert to packed LL' when done */
    cm->final_asis = FALSE ;
    cm->final_super = FALSE ;
    cm->final_ll = TRUE ;
    cm->final_pack = TRUE ;
    cm->final_monotonic = TRUE ;

    /* no need to prune entries due to relaxed supernodal amalgamation, since
     * zeros are dropped with sputil_drop_zeros instead */
    cm->final_resymbol = FALSE ;

    cm->quick_return_if_not_posdef = FALSE;//(nargout < 2) ;

    if(NATURAL_ORDER){
        printf("\n\n NATURALORDER \n\n");
        cm->nmethods = 1 ;
        cm->method [0].ordering = CHOLMOD_NATURAL;
        cm->postorder = FALSE ;
    }
}


PyMODINIT_FUNC
initcholmod(void)
{
    (void) Py_InitModule("cholmod", CholmodMethods);
    import_array();
}


static PyObject *
cholmod_lchol_c(PyObject *self, PyObject *args)
{
    cholmod_common Common;
    cholmod_sparse A, *Lsparse;
    cholmod_factor *L;
    PyObject *Apython,
             *in_shape,
             *Jc,
             *Ir,
             *Adata,
             *nnz;
    PyArrayObject *indices,
                  *indptr,
                  *Ldata,
                  *perm;
    Int nrow, ncol, minor;
    npy_intp dims_output[3];

    cholmod_l_start(&Common);
    sputil_config(&Common);

    Apython = NULL;
    if (!PyArg_ParseTuple(args, "O", &Apython))
        return NULL;

    in_shape = PyObject_GetAttrString(Apython, "shape");
    nrow = PyInt_AsLong(PyTuple_GetItem(in_shape, 0));
    ncol = PyInt_AsLong(PyTuple_GetItem(in_shape, 1));
    Py_DECREF(in_shape);

    Jc = PyObject_GetAttrString(Apython, "indptr");
    Ir = PyObject_GetAttrString(Apython, "indices");
    Adata = PyObject_GetAttrString(Apython, "data");
    nnz = PyObject_GetAttrString(Apython, "nnz");

    A.nzmax = PyInt_AsLong(nnz);
    A.nrow = nrow;
    A.ncol = ncol;
    A.packed = TRUE;
    A.sorted = TRUE;
    A.nz = NULL;
    A.itype = CHOLMOD_LONG;
    A.dtype = CHOLMOD_DOUBLE;
    A.stype = LOWER_TRIANGULAR;
    A.p = (Int *) PyArray_DATA((PyArrayObject*)Jc);
    A.i = (Int *) PyArray_DATA((PyArrayObject*)Ir);
    // XXX empty Apython should be treated separately
    A.x = PyArray_DATA((PyArrayObject*)Adata);
    A.z = NULL;
    A.xtype = CHOLMOD_REAL;

    L = cholmod_l_analyze(&A, &Common);
    cholmod_l_factorize(&A, L, &Common);
    Lsparse = cholmod_l_factor_to_sparse(L, &Common);

    minor = L->minor;
    if (minor < ncol)
    {
        /* remove columns minor to n-1 from Lsparse */
        sputil_trim(Lsparse, minor, &Common) ;
    }

    /* drop zeros from Lsparse */
#ifndef BUG_FOR_BUG_COMPATABILITY_MODE
    cholmod_l_drop(1.0e-6, Lsparse, &Common);
#endif

    dims_output[0] = Lsparse->nzmax;
    dims_output[1] = Lsparse->ncol + 1;
    dims_output[2] = Lsparse->nzmax;
    indices = (PyArrayObject*)PyArray_SimpleNew(1, &(dims_output[0]), NPY_INT64);
    indptr = (PyArrayObject*)PyArray_SimpleNew(1, &(dims_output[1]), NPY_INT64);
    Ldata = (PyArrayObject*)PyArray_SimpleNew(1, &(dims_output[2]), NPY_DOUBLE);
    perm = (PyArrayObject*)PyArray_SimpleNew(1, &ncol, NPY_INT64);

    memcpy(PyArray_DATA(indices), Lsparse->i, PyArray_NBYTES(indices));
    memcpy(PyArray_DATA(indptr), Lsparse->p, PyArray_NBYTES(indptr));
    memcpy(PyArray_DATA(Ldata), Lsparse->x, PyArray_NBYTES(Ldata));
    memcpy(PyArray_DATA(perm), L->Perm, PyArray_NBYTES(perm));

    Py_DECREF(Jc);
    Py_DECREF(Ir);
    Py_DECREF(Adata);
    Py_DECREF(nnz);
    cholmod_l_free_factor(&L, &Common);
    cholmod_l_free_sparse(&Lsparse, &Common);
    cholmod_l_finish(&Common);

    return Py_BuildValue("NNNiN", indices, indptr, Ldata, (minor == ncol) ? 0 : (minor+1), perm);
}
