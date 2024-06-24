function getTensor(type, data, dims) {
    let typedArray;
    if (type === 'bool') {
      return new ort.Tensor(type, [data], [1]);
    } else if (type === 'uint16') {
      typedArray = Uint16Array;
    } else if (type === 'float16') {
      typedArray = Uint16Array;
    } else if (type === 'float32') {
      typedArray = Float32Array;
    } else if (type === 'int32') {
      typedArray = Int32Array;
    } else if (type === 'int64') {
      typedArray = BigInt64Array;
    }
  
    let _data;
    if (Array.isArray(data) || ArrayBuffer.isView(data)) {
      _data = data;
    } else {
      let size = 1;
      dims.forEach((dim) => {
        size *= dim;
      });
      if (data === 'random') {
        _data = typedArray.from({length: size}, () => Math.random());
      } else if (data === 'ramp') {
        _data = typedArray.from({length: size}, (_, i) => i);
      } else {
        _data = typedArray.from({length: size}, () => data);
      }
    }
    return new ort.Tensor(type, _data, dims);
}

async function createFakeSession() {

    const option = {
        executionProviders: [
          {
            name: "webgpu",
            deviceType: "gpu",
          },
        ],
		externalData: ["fakefakefake_t5-encoder-12.onnx.data"],
        graphOptimizationLevel: "all",
      };
	
	const session = await ort.InferenceSession.create('./t5-encoder-12.onnx', option);
	return session;
}

async function createSession() {

    const option = {
        executionProviders: [
          {
            name: "webgpu",
            deviceType: "gpu",
          },
        ],
		externalData: ["t5-encoder-12.onnx.data"],
        graphOptimizationLevel: "all",
      };
	
	const session = await ort.InferenceSession.create('./t5-encoder-12.onnx', option);
	return session;
}

async function run(session) {
	var feeds = {};
    feeds['input_ids'] = getTensor('int64', 99n, [2, 77]);
	const result  = await session.run(feeds);
    const mask = result["hidden_states"]["data"];
	return mask;
}

function start() {
	createFakeSession().then(fake_session => {
        return run(fake_session);
    }).then(fake_mask => {
        console.log(fake_mask);
		console.log('!!!!!!!!!!!!!!!!Completed to run the faked session.!!!!!!!!!!!!!!!!');
    }).then(res1 => {
        return createSession();
    }).then(session => {
        return run(session);
    }).then(mask => {
        console.log(mask);
    }).catch(err => {
    // Handle any reject/error of functions
    });
}

window.onload = start;
