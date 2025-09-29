const createNextPluginPreval = require("next-plugin-preval/config");
const { env } = require("process");
const withNextPluginPreval = createNextPluginPreval();

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  output: "standalone",
};

module.exports = withNextPluginPreval(nextConfig);
