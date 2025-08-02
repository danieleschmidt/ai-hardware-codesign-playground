/**
 * Semantic Release Configuration
 * Automates versioning and changelog generation based on conventional commits
 */

module.exports = {
  branches: [
    'main',
    'next',
    {
      name: 'beta',
      prerelease: true
    },
    {
      name: 'alpha',
      prerelease: true
    }
  ],
  plugins: [
    // Analyze commits to determine release type
    [
      '@semantic-release/commit-analyzer',
      {
        preset: 'conventionalcommits',
        releaseRules: [
          { breaking: true, release: 'major' },
          { revert: true, release: 'patch' },
          { type: 'feat', release: 'minor' },
          { type: 'fix', release: 'patch' },
          { type: 'perf', release: 'patch' },
          { type: 'refactor', release: 'patch' },
          { type: 'docs', release: 'patch' },
          { type: 'test', release: 'patch' },
          { type: 'build', release: 'patch' },
          { type: 'ci', release: 'patch' },
          { type: 'chore', release: 'patch' },
          { type: 'style', release: 'patch' },
          // Hardware-specific release rules
          { type: 'hardware', release: 'minor' },
          { type: 'synthesis', release: 'patch' },
          { type: 'simulation', release: 'patch' },
          { type: 'fpga', release: 'minor' },
          { type: 'asic', release: 'minor' },
          // AI/ML specific release rules
          { type: 'model', release: 'minor' },
          { type: 'training', release: 'patch' },
          { type: 'inference', release: 'patch' },
          { type: 'optimization', release: 'patch' }
        ],
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES', 'BREAKING']
        }
      }
    ],
    // Generate release notes
    [
      '@semantic-release/release-notes-generator',
      {
        preset: 'conventionalcommits',
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES', 'BREAKING']
        },
        writerOpts: {
          commitsSort: ['subject', 'scope']
        },
        presetConfig: {
          types: [
            { type: 'feat', section: '🚀 Features' },
            { type: 'fix', section: '🐛 Bug Fixes' },
            { type: 'perf', section: '⚡ Performance Improvements' },
            { type: 'refactor', section: '♻️ Code Refactoring' },
            { type: 'docs', section: '📚 Documentation' },
            { type: 'test', section: '🧪 Tests' },
            { type: 'build', section: '🏗️ Build System' },
            { type: 'ci', section: '👷 CI/CD' },
            { type: 'chore', section: '🔧 Maintenance' },
            { type: 'style', section: '💄 Styling' },
            // Hardware-specific sections
            { type: 'hardware', section: '🔧 Hardware' },
            { type: 'synthesis', section: '⚙️ Synthesis' },
            { type: 'simulation', section: '🔬 Simulation' },
            { type: 'fpga', section: '🎯 FPGA' },
            { type: 'asic', section: '💎 ASIC' },
            // AI/ML specific sections
            { type: 'model', section: '🤖 Models' },
            { type: 'training', section: '🎓 Training' },
            { type: 'inference', section: '🔮 Inference' },
            { type: 'optimization', section: '🚀 Optimization' }
          ]
        }
      }
    ],
    // Update changelog
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md',
        changelogTitle: '# Changelog\n\nAll notable changes to the AI Hardware Co-Design Playground will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).'
      }
    ],
    // Update package.json version
    [
      '@semantic-release/npm',
      {
        npmPublish: false, // Set to true if publishing to npm
        tarballDir: 'dist'
      }
    ],
    // Update Python package version
    [
      'semantic-release-pypi',
      {
        distDir: 'dist',
        publishRepo: false // Set to true if publishing to PyPI
      }
    ],
    // Create GitHub release
    [
      '@semantic-release/github',
      {
        assets: [
          // Docker images
          {
            path: 'docker-images/*.tar.gz',
            label: 'Docker Images'
          },
          // Python packages
          {
            path: 'dist/*.whl',
            label: 'Python Wheel Package'
          },
          {
            path: 'dist/*.tar.gz',
            label: 'Python Source Distribution'
          },
          // Documentation
          {
            path: 'docs/build/html.zip',
            label: 'Documentation (HTML)'
          },
          // Hardware artifacts
          {
            path: 'hardware/build/synthesis/*.bit',
            label: 'FPGA Bitstreams'
          },
          {
            path: 'hardware/build/reports/*.rpt',
            label: 'Synthesis Reports'
          },
          // Benchmark results
          {
            path: 'benchmarks/results/*.json',
            label: 'Performance Benchmarks'
          }
        ],
        discussionCategoryName: 'Releases',
        addReleases: 'bottom',
        assignees: ['danieleschmidt'],
        releasedLabels: ['released'],
        successComment: false,
        failTitle: false,
        failComment: false
      }
    ],
    // Commit updated files
    [
      '@semantic-release/git',
      {
        assets: [
          'CHANGELOG.md',
          'package.json',
          'package-lock.json',
          'pyproject.toml',
          'docs/VERSION.md'
        ],
        message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
      }
    ]
  ],
  // Global plugin configuration
  preset: 'conventionalcommits',
  tagFormat: 'v${version}',
  repositoryUrl: 'https://github.com/danieleschmidt/ai-hardware-codesign-playground.git',
  // Environment-specific configuration
  ci: true,
  debug: false,
  dryRun: false,
  // Custom configuration for hardware projects
  extends: [
    {
      // Hardware release validation
      verifyConditions: [
        function(pluginConfig, context) {
          // Verify hardware synthesis passes
          if (process.env.VERIFY_SYNTHESIS === 'true') {
            const { execSync } = require('child_process');
            try {
              execSync('make verify-synthesis', { stdio: 'inherit' });
              context.logger.log('✅ Hardware synthesis verification passed');
            } catch (error) {
              throw new Error('❌ Hardware synthesis verification failed');
            }
          }
          
          // Verify ML model validation
          if (process.env.VERIFY_MODELS === 'true') {
            try {
              execSync('python -m pytest tests/models/ -v', { stdio: 'inherit' });
              context.logger.log('✅ ML model validation passed');
            } catch (error) {
              throw new Error('❌ ML model validation failed');
            }
          }
        }
      ]
    }
  ]
};