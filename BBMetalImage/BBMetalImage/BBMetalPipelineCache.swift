import Metal

/// Caches Metal libraries and compute pipeline states by function name.
///
/// Every `BBMetalBaseFilter` used to load the default Metal library and build its own compute
/// pipeline state in `init`. Both are process-wide, immutable results keyed only by bundle and
/// function name, so building them per instance was pure repetition — and building a filter chain
/// creates several filters at once, while attaching a camera consumer builds a chain and a video
/// writer together. Loading a metallib and creating a pipeline state are not cheap, and they land on
/// whichever thread happens to be attaching the camera.
public enum BBMetalPipelineCache {
  private static let lock = NSLock()
  private static var libraries: [Bool: MTLLibrary] = [:]
  private static var pipelines: [String: MTLComputePipelineState] = [:]

  /// The default Metal library for the given bundle, loaded at most once per bundle.
  public static func library(useMainBundle: Bool) -> MTLLibrary? {
    lock.lock()
    defer { lock.unlock() }

    if let cached = libraries[useMainBundle] { return cached }
    guard
      let library = try? BBMetalDevice.sharedDevice.makeDefaultLibrary(
        bundle: useMainBundle ? .main : Bundle.module)
    else { return nil }
    libraries[useMainBundle] = library
    return library
  }

  /// The compute pipeline state for `functionName`, built at most once per function.
  public static func computePipeline(
    functionName: String, useMainBundle: Bool = false
  ) -> MTLComputePipelineState? {
    let key = "\(useMainBundle ? "main" : "module").\(functionName)"

    lock.lock()
    if let cached = pipelines[key] {
      lock.unlock()
      return cached
    }
    lock.unlock()

    // Built outside the lock: creating a pipeline state can take a while, and holding the lock would
    // serialise every other filter being constructed at the same time. Two threads racing on the
    // same function just build it twice and agree on the result.
    guard
      let library = library(useMainBundle: useMainBundle),
      let function = library.makeFunction(name: functionName),
      let pipeline = try? BBMetalDevice.sharedDevice.makeComputePipelineState(function: function)
    else { return nil }

    lock.lock()
    pipelines[key] = pipeline
    lock.unlock()
    return pipeline
  }

  /// Builds the pipeline states for `functionNames` ahead of time, so the first frame through a chain
  /// does not pay for them. Call this off the main thread, before a camera is attached.
  public static func prewarm(functionNames: [String], useMainBundle: Bool = false) {
    for name in functionNames {
      _ = computePipeline(functionName: name, useMainBundle: useMainBundle)
    }
  }
}
