/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      // Proxy everything that should go to XWiki
      {
        source: "/xwiki/:path*",
        destination: "http://159.203.20.200:8080/:path*",
      },
      {
        source: "/bin/:path*",
        destination: "http://159.203.20.200:8080/bin/:path*",
      },
      {
        source: "/resources/:path*",
        destination: "http://159.203.20.200:8080/resources/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
