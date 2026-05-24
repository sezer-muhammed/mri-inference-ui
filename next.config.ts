import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/hf-api/:path*",
        destination:
          "https://sezer-muhammed-mri-inference-api.hf.space/:path*",
      },
    ];
  },
  turbopack: {},
  webpack: (config) => {
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    return config;
  },
};

export default nextConfig;
