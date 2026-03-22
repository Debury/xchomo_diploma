const http = require('http');

// Search for CAMS point
const body = JSON.stringify({
  limit: 5,
  with_payload: true,
  filter: { must: [{ key: "dataset_name", match: { value: "CAMS" } }] }
});

const req = http.request({
  hostname: 'localhost', port: 6333,
  path: '/collections/climate_data/points/scroll',
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
}, res => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    const d = JSON.parse(data);
    console.log('CAMS points:', d.result.points.length);
    d.result.points.forEach(p => {
      console.log('\n--- Point ---');
      console.log(JSON.stringify(p.payload, null, 2));
    });
  });
});
req.write(body);
req.end();
