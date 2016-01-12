#include "fast_fm.h"
#include <glib.h>

typedef struct TestFixture_T {
  cs* X;
  cs* X_t;
  ffm_vector* y;
  ffm_coef* coef;
} TestFixture_T;

void TestFixtureContructorSimple(TestFixture_T* pFixture, gconstpointer pg);

void TestFixtureContructorWide(TestFixture_T* pFixture, gconstpointer pg);

void TestFixtureContructorLong(TestFixture_T* pFixture, gconstpointer pg);

void TestFixtureDestructor(TestFixture_T* pFixture, gconstpointer pg);

TestFixture_T* makeTestFixture(int seed, int n_samples, int n_features, int k);
