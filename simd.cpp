#include <arm_neon.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>
#include <stdexcept>


template <typename T>
class AlignedArray {
public:
    AlignedArray(size_t size, size_t alignment = 16)
        : size_(size), alignment_(alignment) {
        if (alignment_ & (alignment_ - 1))
            throw std::invalid_argument("Alignment must be power of two");
        
        data_ = static_cast<T*>(aligned_alloc(alignment_, size * sizeof(T)));
        if (!data_) throw std::bad_alloc();
    }

    ~AlignedArray() { if (data_) free(data_); }

    T* get() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }

    
    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;

   
    AlignedArray(AlignedArray&& other) noexcept 
        : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            if (data_) free(data_);
            data_ = other.data_;
            size_ = other.size_;
            alignment_ = other.alignment_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
    size_t alignment_ = 16;
};


class Matrix {
public:
    Matrix(int n) : n_(n), data_(n * n, 16) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        
        float* ptr = data_.get();
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                ptr[i * n_ + j] = (i == j) ? 1000.0f * n_ : dist(gen);
            }
        }
    }

    
    float* operator[](int i) noexcept { return data_.get() + i * n_; }
    const float* operator[](int i) const noexcept { return data_.get() + i * n_; }
    int size() const noexcept { return n_; }

    /
    Matrix(const Matrix& other) : n_(other.n_), data_(other.n_ * other.n_, 16) {
        std::copy(other.data_.get(), other.data_.get() + n_ * n_, data_.get());
    }

private:
    int n_;
    AlignedArray<float> data_;
};


void gaussian_elimination_serial(Matrix& A) {
    const int n = A.size();
    for (int k = 0; k < n; ++k) {
        const float diag = A[k][k];
        for (int j = k + 1; j < n; ++j) {
            A[k][j] /= diag;
        }
        A[k][k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            const float factor = A[i][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}


void gaussian_elimination_neon(Matrix& A) {
    const int n = A.size();
    for (int k = 0; k < n; ++k) {
        
        const float diag = A[k][k];
        const float32x4_t inv_diag = vdupq_n_f32(1.0f / diag);

        
        int j = k + 1;
        for (; j <= n - 4; j += 4) {
            float32x4_t row = vld1q_f32(&A[k][j]);
            row = vmulq_f32(row, inv_diag);
            vst1q_f32(&A[k][j], row);
        }
        
        for (; j < n; ++j) {
            A[k][j] /= diag;
        }
        A[k][k] = 1.0f;

        
        for (int i = k + 1; i < n; ++i) {
            const float factor = A[i][k];
            const float32x4_t vfac = vdupq_n_f32(factor);

            j = k + 1;
            for (; j <= n - 4; j += 4) {
                const float32x4_t a_kj = vld1q_f32(&A[k][j]);
                float32x4_t a_ij = vld1q_f32(&A[i][j]);
                a_ij = vmlsq_f32(a_ij, vfac, a_kj);
                vst1q_f32(&A[i][j], a_ij);
            }
            
            for (; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}


bool verify(const Matrix& a, const Matrix& b, float epsilon = 1e-4f) {
    const int n = a.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            if (std::fabs(a[i][j] - b[i][j]) > epsilon) {
                std::cerr << "验证失败 @ (" << i << "," << j << "): "
                          << a[i][j] << " vs " << b[i][j] << "\n";
                return false;
            }
        }
    }
    return true;
}


template <typename Func>
double benchmark(Func&& func, Matrix& m, int warmup = 3, int runs = 5) {
    
    for (int i = 0; i < warmup; ++i) {
        Matrix temp(m);
        func(temp);
    }

    double total = 0;
    for (int i = 0; i < runs; ++i) {
        Matrix temp(m);
        auto start = std::chrono::high_resolution_clock::now();
        func(temp);
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double>(end - start).count();
    }
    return total / runs;
}

int main() {
    try {
        const std::vector<int> sizes = {256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048};
        
        std::cout << "矩阵规模\t串行时间(s)\tNEON时间(s)\t加速比\t验证状态\n";
        std::cout << std::string(60, '-') << "\n";

        for (const auto n : sizes) {
            Matrix base(n);
            Matrix serial_test(base);
            Matrix neon_test(base);

            // 基准测试
            double serial_time = benchmark(gaussian_elimination_serial, serial_test);
            double neon_time = benchmark(gaussian_elimination_neon, neon_test);

            // 验证结果
            const bool valid = verify(serial_test, neon_test);

            // 输出结果
            std::cout << n << "x" << n << "\t"
                      << std::fixed << std::setprecision(3)
                      << serial_time << "\t\t"
                      << neon_time << "\t\t"
                      << std::setprecision(2) << (serial_time / neon_time)
                      << "x\t" << (valid ? "通过" : "失败") << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "错误发生: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}