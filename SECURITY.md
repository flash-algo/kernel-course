# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Security Considerations

kernel-course is primarily an **educational project** with small, self-contained kernels. It does not ship production CUDA/C++ extensions, but exercises and examples may run custom kernels on your GPU when you experiment with Triton or CuTe.

When using this repository:

- Only run code you understand, especially when modifying kernels.
- Use virtual environments to isolate dependencies.
- Be careful when experimenting with very large tensor sizes, as they may cause out-of-memory errors.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

**For security issues:**
- Email: losercheems@gmail.com
- Subject: [SECURITY] kernel_course Vulnerability Report
- Include: Detailed description, reproduction steps, and potential impact

**For general bugs:**
- Use our [GitHub Issues](https://github.com/flash-algo/kernel-course/issues)
- Follow our [contributing guidelines](CONTRIBUTING.md)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution**: Depends on severity and complexity

Critical security issues will be prioritized and may result in emergency releases.

## Security Best Practices

When using kernel-course:

1. **Environment Isolation**
   ```bash
   # Use virtual environments
   python -m venv kernel_course_env
   source kernel_course_env/bin/activate  # Linux/Mac
   # or
   kernel_course_env\Scripts\activate     # Windows
   ```

2. **Dependency Management**
   ```bash
   # Keep dependencies updated
   pip install --upgrade torch kernel-course
   ```

3. **Input Validation**
   ```python
   # Validate tensor shapes and dtypes before processing
   assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
   assert x.shape == y.shape
   ```

4. **Resource Monitoring**
   ```python
   # Monitor GPU memory usage
   import torch
   print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

## Disclosure Policy

- Confirmed vulnerabilities will be disclosed responsibly
- Security fixes will be released as soon as safely possible
- CVE numbers will be requested for significant vulnerabilities
- Credit will be given to security researchers who report issues responsibly

## Contact

For security-related questions or concerns:
- Primary: losercheems@gmail.com
- Project maintainers: See [AUTHORS](AUTHORS) file

For general support:
- GitHub Issues: https://github.com/flash-algo/kernel-course/issues
- Documentation: see the main README and docs/ in this repository.