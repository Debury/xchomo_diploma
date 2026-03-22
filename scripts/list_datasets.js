const http = require('http');

const body = JSON.stringify({
  limit: 200,
  with_payload: { include: ["dataset_name", "hazard_type", "variable", "is_metadata_only"] },
  filter: { must: [{ key: "is_metadata_only", match: { value: true } }] }
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
    const seen = {};
    d.result.points.forEach(p => {
      const k = p.payload.dataset_name;
      if (!(k in seen)) seen[k] = p.payload.hazard_type;
    });
    Object.keys(seen).sort().forEach(k => console.log(k, '|', seen[k]));
    console.log('\nTotal unique datasets:', Object.keys(seen).length);
  });
});
req.write(body);
req.end();
