#!/usr/bin/env node

import {
  runLegacySpine4xOracle,
  writeOracleFailure,
} from './spine4x_legacy_runtime_oracle_core.mjs';

if (process.argv.length !== 4) {
  writeOracleFailure(
    new Error('Usage: node tools/spine42_runtime_oracle.mjs <json-file> <runtime-entry>'),
  );
} else {
  runLegacySpine4xOracle({
    expectedVersion: '4.2.43',
    expectedFamily: '4.2',
    jsonArgument: process.argv[2],
    runtimeArgument: process.argv[3],
  })
    .then((report) => console.info(JSON.stringify(report, null, 2)))
    .catch(writeOracleFailure);
}
